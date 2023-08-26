import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator   

from utility.logger import Logger
import utility.metrics as metrics
from utility.quaternion_distances import quaternion_distance
from models.model import Agent
import models.model as util_model
from dataset.DatasetLidarCam import DatasetKittiRawCalibNet
from dataset.DatasetLidarCam import lidar_project_depth
from dataset.data_utils import (merge_inputs, quaternion_from_matrix)

from environment import environment as env
from environment import transformations as tra
from environment.buffer import Buffer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)


def lidar_project_depth_batch(pc, calib, img_shape):
    depth_img_out = []
    for idx in range(pc.shape[0]):
        depth_img, _ = lidar_project_depth(pc[idx].transpose(0, 1), calib[idx], img_shape)
        depth_img = depth_img.to(DEVICE)
        depth_img_out.append(depth_img)

    depth_img_out = torch.stack(depth_img_out)
    depth_img_out = F.interpolate(depth_img_out, size=[256, 512], mode = 'bilinear', align_corners=False)
    return depth_img_out

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default='/home/zhujt/dataset_zjt/kitti_raw/',
        help="the data path")
    parser.add_argument("--dataset", type=str, default='raw', choices = ['raw', 'odometry', 'raw_calibNet'],
        help="the data type")
    parser.add_argument("--test_type", type=str, default='generalization', choices = ['precision', 'generalization'],
        help="the test type")
    parser.add_argument("--load_model", type=str, default='weights_lstm/raw_1.zip',
        help="the model to load")
    parser.add_argument("--val_sequence", type=list, default=['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync'],
        help="the data for valuation")
    # parser.add_argument("--val_sequence_generalization", type=list, 
    #     default=['2011_09_30_drive_0016_sync', '2011_09_30_drive_0018_sync', '2011_09_30_drive_0020_sync',
    #              '2011_09_30_drive_0028_sync', '2011_09_30_drive_0033_sync', '2011_09_30_drive_0034_sync', 
    #              '2011_09_30_drive_0027_sync'],
    #     help="the data for evaluate the generalization")

    # parser.add_argument("--val_sequence_generalization", type=list, 
    #     default=['2011_09_30_drive_0028_sync'],
    #     help="the data for evaluate the generalization")

    parser.add_argument("--max_t", type=float, default=0.25,
        help="the translation decalibration range")
    parser.add_argument("--max_r", type=float, default=10.,
        help="the rotation decalibration range")
    parser.add_argument("--ITER_EVAL", type=int, default=5,
        help="value iterations")
    parser.add_argument("--batch_size", type=int, default=1,
        help="the batch size for data collection")
    parser.add_argument("--num_worker", type=int, default=5,
        help="the worker nums for training")
    args = parser.parse_args()
    return args

def evaluate(agent, data_path, max_t, max_r, val_sequence):
    args = parse_args()

    dataset_class = DatasetKittiRawCalibNet
    dataset_val = dataset_class(data_path, max_r=max_r, max_t=max_t, split='val',
                                use_reflectance=False, val_sequence=val_sequence)
    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_worker,
                                                # worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)
    print(len(ValImgLoader))

    agent.eval()
    progress = tqdm(BackgroundGenerator(ValImgLoader), total=len(ValImgLoader))

    predictions = []
    with torch.no_grad():
        for data in progress:
            
            rgb_input, depth_input, depth_target, pose_target, pose_source, ds_pc_target, ds_pc_source, calib = env.init(data)
            
            current_source = ds_pc_source
            current_depth = depth_input

            for step in range(args.ITER_EVAL):
                # expert prediction
                if(step == 0):
                    
                    actions, _, hc = agent(rgb_input, current_depth, torch.zeros(2, depth_input.shape[0], 256).to(DEVICE), torch.zeros(2, depth_input.shape[0], 256).to(DEVICE))
                else:
                    actions, _, hc = agent(rgb_input, current_depth, h_last, c_last)

                h_last, c_last = hc[0], hc[1]

                action_t, action_r = actions[0].unsqueeze(1), actions[1].unsqueeze(1)

                action_tr = torch.cat([action_t, action_r], dim = 1)
                new_source, pose_source = env.step_continous(ds_pc_source, action_tr, pose_source)

                current_source = new_source
                current_depth = lidar_project_depth_batch(current_source, calib, (384, 1280))
                current_depth /= 80

            predictions.append(pose_source)

    predictions = torch.cat(predictions)
    eval_metrics, summary_metrics = metrics.compute_stats(predictions, data_loader=ValImgLoader)

    # log test metrics
    print(f"MAE R: {summary_metrics['r_mae']:0.4f}")
    print(f"MAE rr: {summary_metrics['rr_mae']:0.4f}")
    print(f"MAE pp: {summary_metrics['yy_mae']:0.4f}")
    print(f"MAE yy: {summary_metrics['pp_mae']:0.4f}")

    print(f"qdMAE R: {summary_metrics['qd_error']:0.4f}")
    print(f"MAE t: {summary_metrics['t_mae']:0.6f}")
    print(f"MAE x: {summary_metrics['x_mae']:0.6f}")
    print(f"MAE y: {summary_metrics['y_mae']:0.6f}")
    print(f"MAE z: {summary_metrics['z_mae']:0.6f}")

    print(f"ISO R: {summary_metrics['r_iso']:0.4f}")
    print(f"ISO t: {summary_metrics['t_iso']:0.6f}")



if __name__ == '__main__':
    args = parse_args()
    dataset = args.data_folder
    code_path = os.path.dirname(os.path.abspath(__file__))

    pretrain = os.path.join(code_path, args.load_model)
    print("  loading weights...")
    agent = Agent().to(DEVICE)
    if os.path.exists(pretrain):
        util_model.load(agent, pretrain)
    else:
        raise FileNotFoundError(f"No weights found at {pretrain}. Download pretrained weights or run training first.")

    evaluate(agent, dataset, args.max_t, args.max_r, args.val_sequence)



 
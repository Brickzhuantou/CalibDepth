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
        help="the data type") # kitti数据的类型
    parser.add_argument("--use_reflectance", type=bool, default=False,
        help="use reflectance or not")  # 是否使用反射率
    parser.add_argument("--val_sequence", type=list, default=['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync'],
        help="the data for valuation") # 验证集的序列
    parser.add_argument("--val_sequence_generalization", type=list, 
        default=['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync'],
        help="the data for evaluate the generalization")
    parser.add_argument("--max_t", type=float, default=0.2,
        help="the translation decalibration range") # 平移扰动范围的最大值（cm)
    parser.add_argument("--max_r", type=float, default=10.,
        help="the rotation decalibration range") # 旋转扰动范围的最大值（°）
    parser.add_argument("--save_id", type=int, default=1,
        help="the id of the model to be saved") # 模型保存的id
    parser.add_argument("--learning_rate", type=float, default=1e-4,
        help="the learning rate of the optimizer") # 学习率
    parser.add_argument("--learning_rate_step", type=int, default=8,
        help="the learning rate's scale step of the optimizer")
    parser.add_argument("--epoch", type=int, default=50,
        help="the epochs for training")
    parser.add_argument("--batch_size", type=int, default=32,
        help="the batch size for data collection")
    parser.add_argument("--update_batch_size", type=int, default=64,
        help="the batch size for training")
    parser.add_argument("--num_worker", type=int, default=5,
        help="the worker nums for training")
    parser.add_argument("--seed", type=int, default=42,
        help="seeds")
    parser.add_argument("--ITER_TRAIN", type=int, default=5,
        help="train iterations")
    parser.add_argument("--ITER_EVAL", type=int, default=5,
        help="value iterations")
    parser.add_argument("--NUM_TRAJ", type=int, default=4,
        help="trajectory numbers")
    parser.add_argument("--clip_coef", type=float, default=0.2,
        help="clip for policy loss")
    parser.add_argument("--clip_vloss", type=bool, default=True,
        help="clip vloss or not")
    parser.add_argument("--ent_coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    
    # Algorithm specific arguments
    
    args = parser.parse_args()
    return args

class GenerateSeq(nn.Module):
    """
        基于agent网络生成固定长度的标定动作序列
    """
    def __init__(self, agent):
        super(GenerateSeq, self).__init__()
        self.agent = agent
    
    def forward(self, ds_pc_source, calib, depth, rgb, pos_src, pos_tgt, seq_len):
        batch_size = ds_pc_source.shape[0]
        trg_seqlen = seq_len

        # 初始化输出结果
        outputs_save_transl=torch.zeros(batch_size,trg_seqlen,3)
        outputs_save_rot=torch.zeros(batch_size,trg_seqlen,3) # agent动作
        exp_outputs_save_transl=torch.zeros(batch_size,trg_seqlen,3)
        exp_outputs_save_rot=torch.zeros(batch_size,trg_seqlen,3) # 专家监督动作
        h_last = torch.zeros(2, depth.shape[0], 256).to(DEVICE)
        c_last = torch.zeros(2, depth.shape[0], 256).to(DEVICE) # lstm的中间输出
        exp_pos_src = pos_src
        
        # 生成动作序列
        for i in range(0, trg_seqlen):
            # 专家动作
            expert_action = env.expert_step_real(exp_pos_src, pos_tgt)
            # agent动作
            actions, predict_depth, hc = agent(rgb, depth, h_last, c_last)
            h_last, c_last = hc[0], hc[1]
            action_t, action_r = actions[0].unsqueeze(1), actions[1].unsqueeze(1)
            action_tr = torch.cat([action_t, action_r], dim = 1)
            # 下一步状态
            new_source, pos_src = env.step_continous(ds_pc_source, action_tr, pos_src, False)
            exp_new_source, exp_pos_src = env.step_continous(ds_pc_source, expert_action, exp_pos_src, False)
            # 状态更新
            current_source = new_source
            depth = lidar_project_depth_batch(current_source, calib, (384, 1280))  # 更新后点云对应的一个batch的深度图
            depth /= 80
            # 保存
            exp_outputs_save_transl[:,i,:]=expert_action[:,0]
            exp_outputs_save_rot[:,i,:]=expert_action[:,1]
            outputs_save_transl[:,i,:]=actions[0].squeeze(1)
            outputs_save_rot[:,i,:]=actions[1].squeeze(1)
        return exp_outputs_save_transl, exp_outputs_save_rot, outputs_save_transl, outputs_save_rot, pos_src, current_source, predict_depth
            
def train(calib_seq, agent, logger, datapath, max_t, max_r, epochs, batch_size, num_worker,
          lr, lr_step, model_path, val_sequence):
    args = parse_args()
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, 0.5)
    dataset_train = DatasetKittiRawCalibNet(datapath, max_r=max_r, max_t=max_t, split='train',
                                            use_reflectance=False, val_sequence=val_sequence)
    dataset_val = DatasetKittiRawCalibNet(datapath, max_r=max_r, max_t=max_t, split='val',
                                          use_reflectance=False, val_sequence=args.val_sequence_generalization)
    TrainLoader = torch.utils.data.DataLoader(dataset_train, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=num_worker,
                                              pin_memory=True,
                                              drop_last=False,
                                              collate_fn=merge_inputs)
    ValLoader = torch.utils.data.DataLoader(dataset_val, 
                                              batch_size=batch_size, 
                                              shuffle=False, 
                                              num_workers=num_worker,
                                              pin_memory=True,
                                              drop_last=False,
                                              collate_fn=merge_inputs)
    print(len(TrainLoader))
    print(len(ValLoader))
    
    # 初始化
    RANDOM_STATE = np.random.get_state()
    losses_bc, losses_q, losses_tme, losses_pd, loss_depth, losses_all = [], [], [], [], [], []
    episode = 0  # for loss logging (not using epoch)
    best_chamfer = np.infty

    buffer = Buffer()
    buffer.start_trajectory()
    
    cal_loss = torch.nn.SmoothL1Loss(reduction='none')
    FEATURE_loss = util_model.berHuLoss()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        # 训练阶段
        agent.train()
        np.random.set_state(RANDOM_STATE)
        progress = tqdm(BackgroundGenerator(TrainLoader), total=len(TrainLoader))
        for data in progress:
            # 读取batch数据并且初始化
            rgb_input, depth_input, depth_target, pose_target, pose_source, ds_pc_target, ds_pc_source, calib = env.init(data)
            
            current_source = ds_pc_source
            current_depth = depth_input
            
            exp_transl_seq, exp_rot_seq, transl_seq, rot_seq, pos_final, current_source, predict_depth = calib_seq(ds_pc_source, 
                                        calib, current_depth, rgb_input, pose_source, pose_target, args.ITER_TRAIN)
            # 每一步与专家动作之间的平均损失
            loss_translation = cal_loss(transl_seq, exp_transl_seq).sum(2).mean()
            loss_rotation = cal_loss(rot_seq, exp_rot_seq).sum(2).mean()
            clone_loss = loss_rotation + loss_translation 
            
            # 计算最终位姿与真值之间的四元数损失
            R_composed_target = torch.stack([quaternion_from_matrix(pose_target[i, :]) for i in range(pose_target.shape[0])], dim = 0)
            R_composed = torch.stack([quaternion_from_matrix(pos_final[i, :]) for i in range(pos_final.shape[0])], dim = 0)
            qd_error = quaternion_distance(R_composed, 
                            R_composed_target,
                            R_composed.device)
            qd_error = qd_error.abs() * (180.0/np.pi)
            
            # 计算最终位姿和目标之间的平移损失
            t_gt = pose_target[:, :3, 3]
            t_pred = pos_final[:, :3, 3]
            t_mae = torch.abs(t_gt - t_pred).mean(dim=1)
            
            # 点云距离损失
            rand_idxs = np.random.choice(current_source.shape[1], 1024, replace=False)
            src_transformed_samp = current_source[:, rand_idxs, :]
            ref_clean_samp = ds_pc_target[:, rand_idxs, :]
            dist = torch.min(tra.square_distance(src_transformed_samp, ref_clean_samp), dim=-1)[0]
            chamfer_dist = torch.mean(dist, dim=1).view(-1, 1, 1)
            geo_loss = chamfer_dist.mean()

            # 单目深度估计的损失
            mask = depth_target > 0 # 只用大于0的有效值部分进行监督
            depth_loss = FEATURE_loss(predict_depth[mask], depth_target[mask])
            
            # 整体的损失函数
            loss = clone_loss*10 + qd_error.mean()*0.1 + t_mae.mean()*3 + geo_loss * 0.2 + depth_loss * 0.05
            
            # 优化
            optimizer.zero_grad()
            losses_bc.append(10*loss_translation.item())  # 暂时改一下
            losses_q.append(0.1*qd_error.mean().item())
            losses_tme.append(3*t_mae.mean().item())
            losses_pd.append(0.2*geo_loss.item())
            loss_depth.append(depth_loss.item())
            losses_all.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            # 存到log
            logger.record("train/bc", np.mean(losses_bc))
            logger.record("train/q", np.mean(losses_q))
            logger.record("train/tme", np.mean(losses_tme))
            logger.record("train/geo", np.mean(losses_pd))
            logger.record("train/depth", np.mean(loss_depth))
            logger.record("train/all", np.mean(losses_all))
            logger.dump(step=episode)
            
            
            losses_bc, losses_q, losses_tme, losses_pd, loss_depth, losses_all = [], [], [], [], [], []
            episode += 1
        
        scheduler.step()
        RANDOM_STATE = np.random.get_state() 
        
        if ValLoader is not None:
            chamfer_val = evaluate(agent, logger, ValLoader, prefix='val')

        if chamfer_val <= best_chamfer:
            print(f"new best: {chamfer_val}")
            best_chamfer = chamfer_val
            infos = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict()
            }
            # 存储验证集上最优模型
            util_model.save(agent, f"{model_path}.zip", infos)
        logger.dump(step=epoch)
            

def evaluate(agent, logger, loader, prefix='test'):
    agent.eval()
    args = parse_args()
    progress = tqdm(BackgroundGenerator(loader), total=len(loader))
    predictions = []
    
    with torch.no_grad():
        for data in progress:
            rgb_input, depth_input, depth_target, pose_target, pose_source, ds_pc_target, ds_pc_source, calib = env.init(data)
            current_source = ds_pc_source
            current_depth = depth_input

            for step in range(args.ITER_EVAL):
                # 第一步迭代讲h_last和c_last初始化为0
                if(step == 0):
                    actions, _, hc = agent(rgb_input, current_depth, 
                                           torch.zeros(2, depth_input.shape[0], 256).to(DEVICE), 
                                           torch.zeros(2, depth_input.shape[0], 256).to(DEVICE))
                else:
                    actions, _, hc = agent(rgb_input, current_depth, h_last, c_last)
                h_last, c_last = hc[0], hc[1]
                
                action_t, action_r = actions[0].unsqueeze(1), actions[1].unsqueeze(1)
                action_tr = torch.cat([action_t, action_r], dim = 1)
                
                new_source, pose_source = env.step_continous(ds_pc_source, action_tr, pose_source, False)
                current_source = new_source
                current_depth = lidar_project_depth_batch(current_source, calib, (384, 1280))  # 更新后点云对应的一个batch的深度图
                current_depth /= 80
            predictions.append(pose_source)
            
    predictions = torch.cat(predictions)
    _, summary_metrics = metrics.compute_stats(predictions, data_loader=loader)
    logger.record(f"{prefix}/mae-r", summary_metrics['r_mae'])
    logger.record(f"{prefix}/mae-t", summary_metrics['t_mae'])
    logger.record(f"{prefix}/iso-r", summary_metrics['r_iso'])
    logger.record(f"{prefix}/iso-t", summary_metrics['t_iso'])
    logger.record(f"{prefix}/chamfer", summary_metrics['chamfer_dist'])
    logger.record(f"{prefix}/adi-auc", summary_metrics['adi_auc10'] * 100)
    return summary_metrics['chamfer_dist']

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    save_id = args.save_id
    code_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(code_path, "logs")):
        os.mkdir(os.path.join(code_path, "logs"))
    if not os.path.exists(os.path.join(code_path, "weights")):
        os.mkdir(os.path.join(code_path, "weights"))
        
    model_path = os.path.join(code_path, f"weights_lstm/{dataset}_{save_id}")
    logger = Logger(log_dir=os.path.join(code_path, f"logs_lstm/{dataset}/"), log_name=f"calibdepth_{save_id}",
                    reset_num_timesteps=True)
    
    agent = Agent().to(DEVICE)
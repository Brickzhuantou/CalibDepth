#%%
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
from dataset.DatasetLidarCam import lidar_project_depth, get_2D_lidar_projection
from dataset.data_utils import (merge_inputs, quaternion_from_matrix)

from environment import environment as env
from environment import transformations as tra
from environment.buffer import Buffer

import ipcv_utils.utils as plt
import cv2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
def lidar_project_depth_batch(pc, calib, img_shape):
    depth_img_out = []
    for idx in range(pc.shape[0]):
        depth_img, _ = lidar_project_depth(pc[idx].transpose(0, 1), calib[idx], img_shape)
        depth_img = depth_img.to(DEVICE)
        depth_img_out.append(depth_img)

    depth_img_out = torch.stack(depth_img_out)
    depth_img_out = F.interpolate(depth_img_out, size=[256, 512], mode = 'bilinear', align_corners=False)
    return depth_img_out

def get_projected_pts(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.detach().cpu().numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]

    return pcl_uv, pcl_z

def max_normalize_pts(pts):
    return (pts - np.min(pts)) / (np.max(pts) - np.min(pts) + 1e-10)

def get_projected_img(pts, dist, img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dist_norm = max_normalize_pts(dist)*100
    # dist_norm = dist

    for i in range(pts.shape[0]):
        cv2.circle(hsv_img, (int(pts[i, 0]), int(pts[i, 1])), radius=1, color=(int(dist_norm[i]), 255, 255), thickness=-1)

    projection = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    projection = F.interpolate(torch.from_numpy(projection.astype(np.float32)).permute(2,0,1).unsqueeze(0), size=[256, 512], mode = 'bilinear', align_corners=False)
    return projection[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

#%%
# 数据读取
dataset_class = DatasetKittiRawCalibNet
dataset_val = dataset_class('/home/zhujt/dataset_zjt/kitti_raw/', max_r=10., max_t=0.25, split='val', 
                                  use_reflectance=False, val_sequence=['2011_09_26_drive_0020_sync', '2011_09_26_drive_0034_sync'])

ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                 shuffle=True,
                                                 batch_size=1,
                                                 num_workers=4,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)
#%%
# 模型读取
agent = Agent().to(DEVICE)
code_path = '/home/zhujt/code_calib/CalibDepth/'
pretrain = os.path.join(code_path, 'weights_lstm/raw_1.zip')
if os.path.exists(pretrain):
    util_model.load(agent, pretrain)
progress = tqdm(BackgroundGenerator(ValImgLoader), total=len(ValImgLoader))

#%%
target_num = 20
num=0
for data in progress:
    raw_img_path = data['img_path']
    # 读取图片
    rgb_for_show = cv2.imread(raw_img_path[0])
    # rgbforshow转为numpy格式
    rgb_for_show = cv2.cvtColor(rgb_for_show, cv2.COLOR_BGR2RGB)
        
    # rgb_for_show = (data['rgb'][0]*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    item = data['item'][0]
    rgb_input, depth_input, depth_target, pose_target, pose_source, ds_pc_target, ds_pc_source, calib = env.init(data)
    
    current_source = ds_pc_source
    current_depth = depth_input
    
    print(num)
    if num == target_num:
        # 初始的图片存储：
        # plt.imshow(rgb_input[0].permute(1,2,0).cpu())
        # plt.imwrite(rgb_input[0].permute(1,2,0).cpu(), './save_fig/'+str(num)+'_rgb')
        plt.imwrite(rgb_for_show, './save_fig/'+str(num)+'_rgb')
        # plt.imshow((depth_target.expand(-1,3,-1,-1)[0]/80).permute(1,2,0).cpu())
        plt.imwrite((depth_target.expand(-1,3,-1,-1)[0]/80).permute(1,2,0).cpu(), './save_fig/'+str(num)+'_dg')

        # plt.imshow((depth_input/depth_input.max()).expand(-1,3,-1,-1)[0].permute(1,2,0).cpu())
        # plt.imwrite((depth_input/depth_input.max()).expand(-1,3,-1,-1)[0].permute(1,2,0).cpu(), './save_fig/gt'+str(num))
        
        pcl_uv, pcl_z = get_projected_pts(current_source[0].transpose(0,1), calib[0], (384, 1280))
        init_depth = get_projected_img(pcl_uv, pcl_z, np.zeros_like(rgb_for_show))
        # plt.imshow(init_depth)

        init_project_img = get_projected_img(pcl_uv, pcl_z, rgb_for_show)
        # plt.imshow(init_project_img)
        plt.imwrite(init_project_img, './save_fig/'+str(num)+'_iter0')
        print(item)
        pcl_uv, pcl_z = get_projected_pts(ds_pc_target[0].transpose(0,1), calib[0], (384, 1280))
        gt_project_img = get_projected_img(pcl_uv, pcl_z, rgb_for_show)
        # plt.imshow(gt_project_img)
        plt.imwrite(gt_project_img, './save_fig/'+str(num)+'_gt')
        
        for step in range(3):
            # actions, _, action_logprobs, _, value = agent(rgb_input, current_depth)   # 如果是IL单独训练的话，效果应该是看均值的生成效果
            # _, actions, action_logprobs, _, value = agent(rgb_input, current_depth)    # 如果是IL+RL联合进行训练的话，采样输出因为有RL对应损失的监督作用，也可以用来进行测试

            if(step == 0):
                actions, _, hc = agent(rgb_input, current_depth, torch.zeros(2, depth_input.shape[0], 256).to(DEVICE), torch.zeros(2, depth_input.shape[0], 256).to(DEVICE))
            else:
                actions, depth_predict, hc = agent(rgb_input, current_depth, h_last, c_last)
            h_last, c_last = hc[0], hc[1]
            action_t, action_r = actions[0].unsqueeze(1), actions[1].unsqueeze(1)
            action_tr = torch.cat([action_t, action_r], dim = 1)
            
            new_source, pose_source = env.step_continous(ds_pc_source, action_tr, pose_source)
            current_source = new_source
            current_depth = lidar_project_depth_batch(current_source, calib, (384, 1280))  # 更新后点云对应的一个batch的深度图
            current_depth /= 80

            pcl_uv, pcl_z = get_projected_pts(current_source[0].transpose(0,1), calib[0], (384, 1280))
            init_project_img = get_projected_img(pcl_uv, pcl_z, rgb_for_show)
            # plt.imshow(init_project_img)
            plt.imwrite(init_project_img, './save_fig/'+str(num)+'_iter'+str(step+1))
            iter_img = get_projected_img(pcl_uv, pcl_z, np.zeros_like(rgb_for_show))
            
        plt.imwrite((depth_predict.expand(-1,3,-1,-1)[0]/80).permute(1,2,0).detach().cpu().numpy(), './save_fig/'+str(num)+'_pred_d')
        break
    num += 1
    
            
        

# %%

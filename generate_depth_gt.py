# %%
import os
import random
from ip_basic import depth_map_utils
from ip_basic import vis_utils
import torch.nn.functional as F
import numpy as np
import torch
import pykitti
from dataset.DatasetLidarCam import Resampler
import torchvision
import cv2

#%% 点云投影+深度补全
def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z

def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.detach().cpu().numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    # pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1]))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z

    depth_img, process_dict = depth_map_utils.fill_in_multiscale(
        depth_img, extrapolate=False, blur_type='bilateral',
        show_process=False)
    # 添加了深度补全看看效果；

    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    # depth_img = depth_img.permute(2, 0, 1)
    depth_img = depth_img.unsqueeze(0)


    return depth_img, pcl_uv

def lidar_project_depth_batch(pc, calib, img_shape):
    depth_img_out = []
    for idx in range(pc.shape[0]):
        depth_img, _ = lidar_project_depth(pc[idx].transpose(0, 1), calib[idx], img_shape)
        depth_img_out.append(depth_img)

    depth_img_out = torch.stack(depth_img_out)
    depth_img_out = F.interpolate(depth_img_out, size=[256, 512], mode = 'bilinear', align_corners=False)
    return depth_img_out


#%% 数据路径读取
root_dir = '/home/zhujt/dataset_zjt/kitti_raw/' # 数据路径
date = '2011_09_26'  # 以9-26数据为例
dataset_dir = root_dir
seq_list = os.listdir(os.path.join(root_dir, date))

all_files = [] # 用于遍历存储文件路径

for seq in seq_list:
    if not os.path.isdir(os.path.join(dataset_dir, date, seq)):
        continue
    image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
    image_list.sort()
    for image_name in image_list:
        if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
                                            str(image_name.split('.')[0])+'.bin')):
            continue
        if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
                                            str(image_name.split('.')[0])+'.png')): # png
            continue
        all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))
# random.shuffle(all_files)


#%% 遍历投影生成深度图

# 读取标定参数
data = pykitti.raw(root_dir, date, '0001')
calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
            'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}

for item in all_files:
    
    date = str(item.split('/')[0])
    seq = str(item.split('/')[1])
    rgb_name = str(item.split('/')[4])
    # 读取点云数据
    lidar_path = os.path.join(root_dir, date, seq, 'velodyne_points/data', rgb_name+'.bin')
    lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
    pc = lidar_scan.reshape((-1, 4))
    valid_indices = pc[:, 0] < -3.
    valid_indices = valid_indices | (pc[:, 0] > 3.)
    valid_indices = valid_indices | (pc[:, 1] < -3.)
    valid_indices = valid_indices | (pc[:, 1] > 3.)
    pc = pc[valid_indices].copy()
    pc_org = torch.from_numpy(pc.astype(np.float32))

    # 读取标定参数
    RT_cam02 = calib['RT2'].astype(np.float32)
    # camera intrinsic parameter
    calib_cam02 = calib['K2']  # 3x3
    E_RT = RT_cam02
    calib_cal = torch.tensor(calib_cam02, dtype = torch.float) 

    if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
        pc_org = pc_org.t()
    if pc_org.shape[0] == 3:
        homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
        pc_org = torch.cat((pc_org, homogeneous), 0)
    elif pc_org.shape[0] == 4:
        if not torch.all(pc_org[3, :] == 1.):
            pc_org[3, :] = 1.
    else:
        raise TypeError("Wrong PointCloud shape")

    pc_rot = np.matmul(E_RT, pc_org.numpy())
    pc_rot = pc_rot.astype(np.float32).copy()
    pc_in = torch.from_numpy(pc_rot)

    # 对原始点云下采样用于强化学习流程
    transforms = torchvision.transforms.Compose([Resampler(100000)])
    # 为了匹配数据增强的格式引入字典格式
    pc_temp = {'points': pc_in}
    pc_temp['points'] = pc_temp['points'].transpose(0, 1)
    ds_pc = transforms(pc_temp)['points']

    depth_gt = lidar_project_depth_batch(ds_pc.unsqueeze(0), calib_cal.unsqueeze(0), (384, 1280)) 
    if not os.path.exists(os.path.join(root_dir, date, seq, 'depth_gt/data')):
        os.makedirs(os.path.join(root_dir, date, seq, 'depth_gt/data'))
    save_path = os.path.join(root_dir, date, seq, 'depth_gt/data', rgb_name+'.jpg')
    cv2.imwrite(save_path, depth_gt[0].permute(1,2,0).cpu().numpy())  # 存储深度图标签
    
    
#%%

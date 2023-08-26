import os

import mathutils
import csv
import math
from math import radians
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TTF
from PIL import Image


import pykitti
from pykitti import odometry

from utility.utils import ( invert_pose, quaternion_from_matrix, rotate_back, rotate_forward,
                            quaternion_from_matrix, rotate_back )

def get_2D_lidar_projection(pcl, cam_intrinsic):
    """投影点云到图像平面

    Args:
        pcl (_type_): 相机坐标系的点云(3, :)
        cam_intrinsic (_type_): 相机内参
    Returns:
        pcl_uv: 点云的像素坐标
        pcl_z: 点云在每个像素上对应的深度

    """
    pcl_xyz = cam_intrinsic @ pcl
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]
    return pcl_uv, pcl_z
     
def lidar_project_depth(pc, cam_calib, img_shape):
    """获取点云的深度图

    Args:
        pc (_type_): 已经转到相机坐标系的点云
        cam_calib (_type_): 相机内参
        img_shape (_type_): 图像尺寸
    Returns:
        depth_img: 点云的深度图(1, H, W)
        pcl_uv: 点云的像素坐标(N, 2)
    """
    pc = pc[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.detach().cpu().numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc, cam_intrinsic)
    mask = (pcl_uv[:, 0]>0) & (pcl_uv[:, 0]<img_shape[1]) & (
            pcl_uv[:, 1]>0) & (pcl_uv[:, 1]<img_shape[0]) & (
            pcl_z>0)  # 筛选出图像内且深度大于0的点
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.float32)
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.permute(2, 0, 1)
    
    return depth_img, pcl_uv
    
class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            sample['points_src'] = self._resample(sample['points_src'], src_size)
            sample['points_ref'] = self._resample(sample['points_ref'], ref_size)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]
     
    

class DatasetKittiRawCalibNet(Dataset):
    def __init__(self, dataset_dir, transform = None, augmentation = False,
                 use_reflectance = False, max_t = 1.5, max_r = 15.0,
                 split = 'val', device = 'cpu',
                 val_sequence = ['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync']):
        super(DatasetKittiRawCalibNet, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.max_depth = 80
        self.K_list = {}
        
        self.all_files = []
        date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        data_drive_list = ['0001', '0002', '0004', '0016', '0027']
        self.calib_date = {}
        
        # 获取不同日期对应的calib文件
        for i in range(len(date_list)):       # 这个循环应该是为了获得不同日期的数据的calib文件
            date = date_list[i]
            data_drive = data_drive_list[i]
            data = pykitti.raw(self.root_dir, date, data_drive)
            calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
                     'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}
            self.calib_date[date] = calib
            
        date = val_sequence[0][:10]
        test_list = ['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync', '2011_10_03_drive_0027_sync']
        seq_list = os.listdir(os.path.join(self.root_dir, date))
        
        # 读取预先存储的训练集和测试集文件名
        train_path = os.path.join(self.root_dir, 'train.txt')
        test_path = os.path.join(self.root_dir, 'test.txt')

        self.train_array = np.loadtxt(train_path, dtype=str)
        self.test_array = np.loadtxt(test_path, dtype=str)
        
        
    def custom_transform(self, rgb, img_rotation, h_mirror, flip = False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        if self.split == 'train':
            return len(self.train_array)
        else:
            return len(self.test_array)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            item = self.train_array[idx]
        else:
            item = self.test_array[idx]
        
        # 路径获取
        date = str(item.split('/')[0])
        seq = str(item.split('/')[1])
        rgb_name = str(item.split('/')[4])
        img_path = os.path.join(self.root_dir, date, seq, 'image_02/data', rgb_name+'.png')
        lidar_path = os.path.join(self.root_dir, date, seq, 'velodyne_points/data', rgb_name+'.bin')
        
        # 数据获取
        lidar_scan = np.fromfile(lidar_path, dtype = np.float32)
        pc = lidar_scan.reshape(-1, 4)
        valid_indices = pc[:,0] <  -3.
        valid_indices = valid_indices | (pc[:,0] >  3.)
        valid_indices = valid_indices | (pc[:,1] < -3.)
        valid_indices = valid_indices | (pc[:,1] >  3.)
        pc = pc[valid_indices].copy() # 滤除自车
        
        pc_org = torch.from_numpy(pc.astype(np.float32))
        
        if self.use_reflectance:
            reflectence = pc[:,3].copy()
            reflectence = torch.from_numpy(reflectence).float()
        
        # 读取标定参数；
        calib = self.calib_date[date]
        RT_cam02 = calib['RT2'].astype(np.float32)
        calib_cam02 = calib['K2']
        
        # 校验点云数据，保证输出为4xN，且最后一行为1
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
        
        # 转到相机坐标系下
        pc_rot = np.matmul(RT_cam02, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)
        
        # 图像数据获取
        img = Image.open(img_path)
        img_rotation = 0.
        h_mirror = False
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)
        
        # 添加扰动
        max_angle = self.max_r
        rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
        roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
        rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
        transl_x = np.random.uniform(-self.max_t, self.max_t)
        transl_y = np.random.uniform(-self.max_t, self.max_t)
        transl_z = np.random.uniform(-self.max_t, self.max_t)
        initial_RT = 0.0
            
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))
        
        R, T = invert_pose(R,T) # 计算求逆后的四元数和平移向量
        R, T = torch.tensor(R), torch.tensor(T)
        
        calib = calib_cam02
        if h_mirror:
            calib[2] = img.shape[2] - calib[2]
        calib = torch.tensor(calib, dtype=torch.float32)
        
        # 获取点云深度图
        max_depth = 80.
        real_shape = [img.shape[1], img.shape[2], img.shape[0]]
        
        # 点云下采样
        transformes = torchvision.transforms.Compose(
                    [Resampler(100000)]) # 对点云下采样为100000个点
        pc_temp = {'points': pc_in}
        pc_temp['points'] = pc_temp['points'].transpose(0, 1)
        ds_pc = transformes(pc_temp)['points'].transpose(0, 1)
        
        # 获取深度图标签
        depth_path = os.path.join(self.root_dir, date, seq, 'depth_gt/data', rgb_name+'.jpg')
        depth_gt = Image.open(depth_path)
        to_tensor = torchvision.transforms.ToTensor()
        depth_gt = to_tensor(depth_gt)*255
        
        # 获取扰动的点云深度图
        R_m = mathutils.Quaternion(R).to_matrix()
        R_m.resize_4x4()
        T_m = mathutils.Matrix.Translation(T)
        RT_m = T_m * R_m
        
        pc_rotated = rotate_back(pc_in, RT_m) # Pc’ = RT * Pc
        ds_pc_rotated = rotate_back(ds_pc, RT_m)   # 下采样后旋转扰动的点云
        
        depth_img, uv = lidar_project_depth(ds_pc_rotated, calib, real_shape)
        depth_img /= max_depth # 深度值归一化
        
        # 点云数据
        pc_target = pc_in  # 原始的相机坐标系的点云为target
        pc_source = pc_rotated # 扰动后的点云为source
        
        # 位姿数据（扰动的逆作为点云位姿）
        i_pose_target = np.array(RT_m, dtype=np.float32)
        pose_target = i_pose_target.copy()
        pose_target[:3, :3] = pose_target[:3, :3].T
        pose_target[:3, 3] = -np.matmul(pose_target[:3, :3], pose_target[:3, 3])
        pose_target = torch.from_numpy(pose_target)
        
        pose_source = torch.eye(4)
        
        # 数据统一字典格式输出
        if self.split == 'test':
            sample = {'rgb': img,
                      'calib': calib,
                      'rgb_name': rgb_name + '.png', 
                      'item': item, 'extrin': RT_cam02,
                      'tr_error': T, 'rot_error': R, 
                      'img_path': img_path,
                      'initial_RT': initial_RT,
                      'pc_target': pc_target,
                      'pc_source': pc_source,
                      'pose_target': pose_target,
                      'pose_source': pose_source,
                      'ds_pc_target': ds_pc, 
                      'ds_pc_source': ds_pc_rotated,
                      'depth_gt': depth_gt, 
                      'depth_img': depth_img,
                      }
        else:
            sample = {'rgb': img,
                      'calib': calib,
                      'rgb_name': rgb_name, # TODO：少了个后缀验一下
                      'item': item, 
                      'img_path': img_path,
                      'tr_error': T, 'rot_error': R, 
                      'pc_target': pc_target,
                      'pc_source': pc_source,
                      'pose_target': pose_target,
                      'pose_source': pose_source,
                      'ds_pc_target': ds_pc, 
                      'ds_pc_source': ds_pc_rotated,
                      'depth_gt': depth_gt, 
                      'depth_img': depth_img,
                      }
            
        return sample
        
        
        
        



from distutils.command.build_scripts import first_line_re
import sys
sys.path.append('/home/zhujt/code_calib/CalibDepth')
import pytest
import torch
import tqdm
from prefetch_generator import BackgroundGenerator
from dataset.DatasetLidarCam import DatasetKittiRawCalibNet as DatasetKittiRawCalibNet
from dataset.DatasetLidarCam import lidar_project_depth, Resampler
from dataset.data_utils import merge_inputs
from environment import environment as env
import pykitti
import os
import numpy as np
import torchvision
from torchvision import transforms as transf
from PIL import Image
import cv2


def test_data_init():
    """验证数据处理后的维度是否符合模型的输入要求"""
    # 路径获取
    root_dir = '/home/zhujt/dataset_zjt/kitti_raw/'
    date = '2011_09_26'
    dataset_dir = root_dir
    seq_list = os.listdir(os.path.join(root_dir, date))
    seq = seq_list[0]
    image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
    image_name = image_list[0]
    item = os.path.join(date, seq, 'image_02/data', image_name.split('.')[0])
    data = pykitti.raw(root_dir, date, '0001')
    calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
                'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}
    date = str(item.split('/')[0])
    seq = str(item.split('/')[1])
    rgb_name = str(item.split('/')[4])

    # 读取图像数据
    img_path = os.path.join(root_dir, date, seq, 'image_02/data', rgb_name+'.png') # png
    img = Image.open(img_path)
    to_tensor = transf.ToTensor()
    img = to_tensor(img)
    real_shape = [img.shape[1], img.shape[2], img.shape[0]]

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

    transforms = torchvision.transforms.Compose([Resampler(100000)])   #采样多少个点
    pc_temp = {'points': pc_in}
    pc_temp['points'] = pc_temp['points'].transpose(0, 1)
    ds_pc = transforms(pc_temp)['points'].transpose(0, 1)
    
    depth_img, uv = lidar_project_depth(ds_pc, calib_cal, real_shape)
    
    assert depth_img.shape[1] == real_shape[0] and depth_img.shape[2] == real_shape[1]
    
    # 存储原图以及深度图到当前路径
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0)) * 255

    depth_img = depth_img.numpy()
    depth_img = np.transpose(depth_img, (1, 2, 0)) * 255
    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=2)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR)
    # print(img.shape)
    # print(depth_img.shape)

    # 存储图片看投影效果；
    cv2.imwrite('/home/zhujt/code_calib/CalibDepth/test/test.png', img)
    cv2.imwrite('/home/zhujt/code_calib/CalibDepth/test/test_depth.png', depth_img)
    
        
    


if __name__ == 'main':
    test_data_init()
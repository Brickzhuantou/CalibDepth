import sys
sys.path.append('/home/zhujt/code_calib/CalibDepth')
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import torch
import environment.transformations as tra
from environment import environment as env

from utility.utils import quaternion_from_matrix
from utility.quaternion_distances import quaternion_distance


def compute_metrics(pose_target, ds_pc_source, ds_pc_target, pred_transforms):
    gt_transforms = pose_target
    
    igt_transforms = torch.eye(4, device=pred_transforms.device).repeat(gt_transforms.shape[0], 1, 1)
    igt_transforms[:, :3, :3] = gt_transforms[:, :3, :3].transpose(2, 1)
    igt_transforms[:, :3, 3] = -(igt_transforms[:, :3, :3] @ gt_transforms[:, :3, 3].view(-1, 3, 1)).view(-1, 3)
    
    points_src = ds_pc_source[..., :3]
    points_ref = ds_pc_target[..., :3]
    points_raw = points_ref

    # 四元数评测指标
    R_composed_target = torch.stack([quaternion_from_matrix(gt_transforms[i, :]) 
                                     for i in range(gt_transforms.shape[0])], dim = 0)
    R_composed = torch.stack([quaternion_from_matrix(pred_transforms[i, :]) 
                              for i in range(pred_transforms.shape[0])], dim = 0)
    
    qd_error = quaternion_distance(R_composed, 
                    R_composed_target,
                    R_composed.device)
    qd_error = qd_error.abs() * (180.0/np.pi)
    
    # 欧拉角评测指标
    r_gt_euler_deg = np.stack([Rotation.from_matrix(r.cpu().numpy()).as_euler('xyz', degrees=True)
                               for r in gt_transforms[:, :3, :3]])
    r_pred_euler_deg = np.stack([Rotation.from_matrix(r.cpu().numpy()).as_euler('xyz', degrees=True)
                                 for r in pred_transforms[:, :3, :3]])
    t_gt = gt_transforms[:, :3, 3]
    t_pred = pred_transforms[:, :3, 3]
    r_mae = np.abs(r_gt_euler_deg - r_pred_euler_deg).mean(axis=1)
    t_mae = torch.abs(t_gt - t_pred).mean(dim=1) # 分别计算旋转平移三个维度的平均误差
    
    # 计算iso误差指标
    concatenated = igt_transforms @ pred_transforms
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    r_iso = torch.rad2deg(torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)))
    t_iso = concatenated[:, :3, 3].norm(dim=-1)
    
    # 旋转平移六个维度各自的平均误差
    xyz_mae = torch.abs(t_gt - t_pred)
    x_mae, y_mae, z_mae = xyz_mae[:,0], xyz_mae[:,1], xyz_mae[:,2]
    rpy_mae = np.abs(np.stack([Rotation.from_matrix(r.cpu().numpy()).as_euler('xyz', degrees=True)
                                 for r in concatenated[:, :3, :3]]))
    rr_mae, pp_mae, yy_mae = rpy_mae[:,0], rpy_mae[:,1], rpy_mae[:,2]
    
    # 点云距离指标
    src_transformed = (pred_transforms[:, :3, :3] @ points_src.transpose(2, 1)).transpose(2, 1)\
                      + pred_transforms[:, :3, 3][:, None, :] # 用预测值校正后的点云
    
    rand_idxs = np.random.choice(src_transformed.shape[1], 1024, replace=False) # 随机采样1024个点
    src_transformed_samp = src_transformed[:, rand_idxs, :]
    points_ref_samp = points_ref[:, rand_idxs, :] # 分别对原始点云和参考真值点云进行采样
    
    dist_src = torch.min(tra.square_distance(src_transformed_samp, points_ref_samp), dim=-1)[0]
    dist_ref = torch.min(tra.square_distance(points_ref_samp, src_transformed_samp), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1) # 计算倒角距离
    
    metrics = {
        'r_mae': r_mae,
        'rr_mae': rr_mae,
        'pp_mae': pp_mae,
        'yy_mae': yy_mae,
        'qd_error': qd_error.cpu().numpy(),
        't_mae': t_mae.cpu().numpy(),
        'x_mae': x_mae.cpu().numpy(),
        'y_mae': y_mae.cpu().numpy(),
        'z_mae': z_mae.cpu().numpy(),
        'r_iso': r_iso.cpu().numpy(),
        't_iso': t_iso.cpu().numpy(),
        'chamfer_dist': chamfer_dist.cpu().numpy()
    }
    return metrics

def summarize_metrics(metrics):
    summarized = {}
    for k in metrics:
        metrics[k] = np.hstack(metrics[k])
        summarized[k] = np.mean(metrics[k])
    return summarized

def compute_stats(pred_transforms, data_loader):
    metrics_for_iter = defaultdict(list)
    num_processed = 0
    with torch.no_grad():
        for data in tqdm(data_loader, leave=False):
            dict_all_to_device(data, pred_transforms.device)
            _,_,_, pose_target, pose_source, ds_pc_target, ds_pc_source, calib = env.init(data)
            batch_size = pose_source.shape[0]
            cur_pred_transforms = pred_transforms[num_processed:num_processed+batch_size]
            metrics = compute_metrics(pose_target, ds_pc_source, ds_pc_target, cur_pred_transforms)
            for k in metrics:
                metrics_for_iter[k].append(metrics[k])
            num_processed += batch_size
            
    summary_metrics = summarize_metrics(metrics_for_iter)
    return metrics_for_iter, summary_metrics

def dict_all_to_device(tensor_dict, device):
    """Sends everything into a certain device
    via RPMNet """
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].to(device)
            if tensor_dict[k].dtype == torch.double:
                tensor_dict[k] = tensor_dict[k].float()
        if isinstance(tensor_dict[k], dict):
            dict_all_to_device(tensor_dict[k], device)
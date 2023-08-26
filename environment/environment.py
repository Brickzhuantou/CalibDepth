import environment.transformations as tra
import torch.nn.functional as F
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init(data):
    """数据进行预处理格式化

    Args:
        data (_type_): _description_
    """
    img_shape = (384, 1280) # 目标图像尺寸
    
    pose_target = data['pose_target'].to(DEVICE)
    pose_source = data['pose_source'].to(DEVICE)
    
    ds_pc_target_input = []
    ds_pc_source_input = []
    calib_input = []
    depth_img_input = []
    depth_gt_input = []
    rgb_input = []
    
    for idx in range(len(data['rgb'])):
        rgb = data['rgb'][idx].cuda()
        depth_gt = data['depth_gt'][idx].cuda()
        depth_img = data['depth_img'][idx].cuda()
        ds_pc_source = data['ds_pc_source'][idx].cuda()
        ds_pc_target = data['ds_pc_target'][idx].cuda()
        calib = data['calib'][idx].cuda()
        
        shape_pad = [0, 0, 0, 0]
        shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
        shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1
        
        rgb = F.pad(rgb, shape_pad)
        depth_img = F.pad(depth_img, shape_pad) # 填充为目标尺寸
        
        rgb_input.append(rgb)
        depth_img_input.append(depth_img)
        depth_gt_input.append(depth_gt)

        ds_pc_target_input.append(ds_pc_target.transpose(0, 1)[:, :3])
        ds_pc_source_input.append(ds_pc_source.transpose(0, 1)[:, :3])    # （4xN->Nx3)
        calib_input.append(calib)
        
    depth_img_input = torch.stack(depth_img_input)
    depth_gt_input = torch.stack(depth_gt_input)
    rgb_input = torch.stack(rgb_input)
    ds_pc_source_input = torch.stack(ds_pc_source_input)
    ds_pc_target_input = torch.stack(ds_pc_target_input)
    calib_input = torch.stack(calib_input)
    
    rgb_input = F.interpolate(rgb_input, size=[256, 512], mode = 'bilinear', align_corners=False)
    depth_img_input = F.interpolate(depth_img_input, size=[256, 512], mode = 'bilinear', align_corners=False)
    
    return rgb_input, depth_img_input, depth_gt_input, pose_target, pose_source, ds_pc_target_input, ds_pc_source_input, calib_input


def step_continous(source, actions, pose_source):
    """
    Update the state (source and accumulator) using the given actions.
    """
    steps_t, steps_r = actions[:, 0], actions[:, 1]
    pose_update = torch.eye(4, device=DEVICE).repeat(pose_source.shape[0], 1, 1)
    pose_update[:, :3, :3] = tra.axis_angle_to_matrix(steps_r)
    pose_update[:, :3, 3] = steps_t
    pose_source = pose_update @ pose_source.to(DEVICE) 
    new_source = tra.apply_trafo(source.to(DEVICE), pose_source, False)
    
    return new_source, pose_source

    
def expert_step_real(pose_source, targets, mode='steady'):  
    """
    Get the expert action in the current state. 直接输出当前的source和target的偏差作为专家动作
    """
    delta_t = targets[:, :3, 3] - pose_source[:, :3, 3]
    delta_R = targets[:, :3, :3] @ pose_source[:, :3, :3].transpose(2, 1)  # global accumulator

    delta_r = tra.matrix_to_axis_angle(delta_R)       # 旋转矩阵到旋转向量

    steps_t = delta_t.unsqueeze(1)
    steps_r = delta_r.unsqueeze(1)
    action = torch.cat([steps_t, steps_r], dim=1)
    
    return action



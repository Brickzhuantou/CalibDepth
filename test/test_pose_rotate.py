import sys
sys.path.append('/home/zhujt/code_calib/CalibDepth')
import pytest
import torch
import mathutils
import numpy as np

from environment import environment as env
from utility.utils import ( invert_pose, quaternion_from_matrix, rotate_back, rotate_forward,
                            quaternion_from_matrix, rotate_back )

def test_rotate():
    """测试点云的扰动+恢复
    """
    # 创建一个测试用的点云 PC，假设是 4xN 的形状
    pc_target = torch.tensor([[1.0, 2.0, 3.0, 1.0],
                       [4.0, 5.0, 6.0, 1.0],
                       [7.0, 8.0, 9.0, 1.0],
                       [10.0, 11.0, 12.0, 1.0],
                       [10.0, 11.0, 12.0, 1.0]])
    
    # 添加扰动
    max_angle = 20
    max_t = 1.5
    rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, max_t)
    initial_RT = 0.0
        
    R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
    T = mathutils.Vector((transl_x, transl_y, transl_z))
    
    R_m = mathutils.Quaternion(R).to_matrix()
    R_m.resize_4x4()
    T_m = mathutils.Matrix.Translation(T)
    RT_m = T_m * R_m

    pc_rotated = rotate_back(pc_target, RT_m) # Pc’ = RT * Pc
    pc_source = pc_rotated
    
    # 位姿数据（扰动的逆作为点云位姿）
    i_pose_target = np.array(RT_m, dtype=np.float32)
    pose_target = i_pose_target.copy()
    pose_target[:3, :3] = pose_target[:3, :3].T
    pose_target[:3, 3] = -np.matmul(pose_target[:3, :3], pose_target[:3, 3])
    pose_target = torch.from_numpy(pose_target)
    pose_source = torch.eye(4)
    
    # 计算专家动作
    expert_action = env.expert_step_real(pose_source.unsqueeze(0), pose_target.unsqueeze(0), False)
    new_pc_source, pos_src = env.step_continous(pc_source.unsqueeze(0), expert_action, pose_source.unsqueeze(0))
    
    
    print("pc_target: ", pc_target)
    print("pc_source: ", pc_source)
    print("new_pc_source: ", new_pc_source)
    assert torch.allclose(pc_target.to("cuda"), new_pc_source) # 恢复后的点云与原始点云一致
    assert torch.allclose(pose_target.to("cuda"), pos_src) # 恢复后的位姿与原始位姿一致
    
    
if __name__ == '__main__':
    test_rotate()
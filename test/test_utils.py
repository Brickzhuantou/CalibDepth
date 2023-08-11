import sys
sys.path.append('/home/zhujt/code_calib/CalibDepth')
import torch
import mathutils
import pytest
# import utility.utils as utils
from utility import utils

import unittest
from utility.utils import rotate_forward
from mathutils import Matrix, Euler

def test_rotate_forward():
    # 创建一个测试用的点云 PC，假设是 4xN 的形状
    PC = torch.tensor([[1.0, 2.0, 3.0, 1.0],
                       [4.0, 5.0, 6.0, 1.0],
                       [7.0, 8.0, 9.0, 1.0],
                       [10.0, 11.0, 12.0, 1.0]])

    # 创建测试用的旋转矩阵 R，这里用单位矩阵作为测试
    R = torch.eye(4)

    # 创建测试用的平移向量 T，假设是 [1, 2, 3]
    # T = torch.tensor([1.0, 2.0, 3.0])
    T = None

    # 调用 rotate_forward 函数进行旋转
    rotated_PC = rotate_forward(PC, R, T)

    # 进行断言，比较旋转后的点云 rotated_PC 是否与预期一致
    # expected_rotated_PC = torch.tensor([[ 6.0,  7.0,  8.0, 1.0],
    #                                     [11.0, 12.0, 13.0, 1.0],
    #                                     [16.0, 17.0, 18.0, 1.0],
    #                                     [21.0, 22.0, 23.0, 1.0]])
    expected_rotated_PC = PC
    assert torch.allclose(rotated_PC, expected_rotated_PC)

    # 添加更多测试用例，如测试不同的旋转角度、平移向量等情况
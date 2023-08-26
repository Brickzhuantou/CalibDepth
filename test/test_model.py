import sys
sys.path.append('/home/zhujt/code_calib/CalibDepth')
import pytest
import torch
from models.model import Agent

@pytest.fixture
def sample_input():
    batch_size = 2
    channels = 3
    height = 256
    width = 512
    h_n = torch.zeros(2, batch_size, 256)
    c_n = torch.zeros(2, batch_size, 256)

    rgb_img = torch.rand(batch_size, channels, height, width)
    depth_img = torch.rand(batch_size, 1, height, width)

    return rgb_img, depth_img, h_n, c_n


def test_agent_forward(sample_input):
    rgb_img, depth_img, h_n, c_n = sample_input
    agent = Agent()

    action_mean, predict_depth, hc = agent(rgb_img, depth_img, h_n, c_n)

    assert isinstance(action_mean, list)
    assert len(action_mean) == 2
    assert action_mean[0].shape == (rgb_img.shape[0], 3)
    assert action_mean[1].shape == (rgb_img.shape[0], 3)

    assert predict_depth.shape == depth_img.shape

    assert hc[0].shape == h_n.shape
    assert hc[1].shape == c_n.shape

if __name__ == '__main__':
    pytest.main()


# 主要用于测试模型的前向推理过程，测试模型的输入输出是否符合预期
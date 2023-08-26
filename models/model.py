from re import M
from turtle import forward
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F

# depth
import models.base.resnet_encoder as resnet_encoder
from models.base.R_MSFM import R_MSFM3

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def MyConv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.ReLU())


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        # nn.LeakyReLU(0.1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0), nn.ReLU())



# 特征提取部分
class StateEmbed(nn.Module):
    def __init__(self):
        super(StateEmbed, self).__init__()
        
        # 图像分支
        self.resnet_encoder = resnet_encoder.ResnetEncoder(18, True)
        self.rgb_depth_decoder = R_MSFM3(False)
        self.rgb_dep_encoder = nn.Sequential(  # 对图像分支深度图编码    
            MyConv(1,16,8,4,2),
            MyConv(16,32,4,2,1),
        )
        self.rgb_res_encoder = MyConv(128, 32,  # 用于残差特征提
            kernel_size=1, padding=0, stride=1)

        # 点云深度图分支
        self.depth_backbone = resnet_encoder.ResnetEncoder(18, True)
        self.depth_encoder = MyConv(128,32,kernel_size=1,padding=0,stride=1) # 对点云分支深度图编码
        
        # 特征融合
        self.match_layer = nin_block(64, 32, 8, 4, 2)
        
        self.match_block = nn.Sequential(
            MyConv(32, 64, kernel_size=1, padding=0, stride=1),
            MyConv(64, 64, kernel_size=3, padding=1, stride=1),
            MyConv(64, 32, kernel_size=1, padding=0, stride=1)
        )
        self.leakyRELU = nn.LeakyReLU(0.1)
        
    def forward(self, rgb_img, depth_img):
        # 图像分支
        res_emb = self.resnet_encoder(rgb_img)
        rgb_dep = self.rgb_depth_decoder(res_emb)[("disp_up", 2)]
        _, scale_dep = disp_to_depth(rgb_dep, 0.1, 80) # 获取中间深度图
        scale_dep_emb = self.rgb_dep_encoder(scale_dep/80) # 图像分支深度图编码
        _, _, x3 = res_emb
        rgb_ori_emb = self.rgb_res_encoder(x3) # 用于残差特征提取
        rgb_emb = torch.add(rgb_ori_emb, scale_dep_emb) # 残差特征提取与深度图编码融合
        
        
        # 点云分支
        _, _, depth_emb = self.depth_backbone(depth_img.expand(-1,3,-1,-1))
        depth_emb = self.depth_encoder(depth_emb)
        
        # 特征融合
        match_emb = torch.cat((rgb_emb, depth_emb), dim=1)
        match_emb = self.match_layer(match_emb)    
        match_emb = match_emb + self.match_block(match_emb)
        match_emb = self.leakyRELU(match_emb)
        
        return match_emb, scale_dep # 返回融合特征和中间深度图
    
    
# 标定动作预测部分
class CalibActionHead(nn.Module):
    def __init__(self):
        super(CalibActionHead, self).__init__()
        self.activation = nn.ReLU()
        self.input_dim = 32*8*16
        self.head_dim = 128
        
        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = 2*self.head_dim, 
                            num_layers = 2, batch_first = True, dropout = 0.5)

        self.emb_r = nn.Sequential(
            nn.Linear(self.head_dim*2, self.head_dim),
            self.activation
        )

        self.emb_t = nn.Sequential(
            nn.Linear(self.head_dim*2, self.head_dim),
            self.activation
        )
        self.action_t = nn.Linear(self.head_dim, 3)
        self.action_r = nn.Linear(self.head_dim, 3)
        
    def forward(self, state, h_n, c_n):
        state = state.view(state.shape[0], -1)

        output, (h_n, c_n) = self.lstm(state.unsqueeze(1), (h_n, c_n))
        emb_t = self.emb_t(output)
        emb_r = self.emb_r(output)
        
        action_mean_t = self.action_t(emb_t).squeeze(1)
        action_mean_r = self.action_r(emb_r).squeeze(1)
        action_mean = [action_mean_t, action_mean_r]
        
        return action_mean, (h_n, c_n)
    
    
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.state_emb = StateEmbed()
        self.calib_action = CalibActionHead()
    def forward(self, rgb_img, depth_img, h_last, c_last):
        state_emb, predict_depth = self.state_emb(rgb_img, depth_img)
        action_mean, hc = self.calib_action(state_emb, h_last, c_last)

        return action_mean, predict_depth, hc

class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(torch.abs(pred - target))
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss
    
    
# --- model helpers
def load(model, path):
    infos = torch.load(path)
    model.load_state_dict(infos['model_state_dict'])
    return infos


def save(model, path, infos={}):
    infos['model_state_dict'] = model.state_dict()
    torch.save(infos, path)
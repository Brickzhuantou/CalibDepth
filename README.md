

# CalibDepth

— LiDAR-Camera online calibration is of great significance for building a stable autonomous driving perception
system. For online calibration, a key challenge lies in constructing a unified and robust representation between multimodal sensor data. Most methods extract features manually
or implicitly with an end-to-end deep learning method. The
former suffers poor robustness, while the latter has poor
interpretability. In this paper, we propose CalibDepth, which
uses depth maps as the unified representation for image and
LiDAR point cloud. CalibDepth introduces a sub-network for
monocular depth estimation to assist online calibration tasks. To
further improve the performance, we regard online calibration
as a sequence prediction problem, and introduce global and
local losses to optimize the calibration results. CalibDepth shows
excellent performance in different experimental setups.

<!-- PROJECT SHIELDS -->

<!-- 
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->
<!-- <br />

<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/">
    <img src="images/Graphic abstract.jpg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">"完美的"README模板</h3>
  <p align="center">
    一个"完美的"README模板去快速开始你的项目！
    <br />
    <a href="https://github.com/shaojintian/Best_README_template"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/shaojintian/Best_README_template">查看Demo</a>
    ·
    <a href="https://github.com/shaojintian/Best_README_template/issues">报告Bug</a>
    ·
    <a href="https://github.com/shaojintian/Best_README_template/issues">提出新特性</a>
  </p>

</p> -->


 论文链接：https://ieeexplore.ieee.org/document/10161575
 
## 目录

- [环境配置](#环境配置)
- [文件目录说明](#文件目录说明)
- [数据准备](#数据准备)
- [运行](#运行)
- [鸣谢](#鸣谢)

### 环境配置


1. 创建虚拟环境（python 3.6.13）
2. Clone the repo
```sh
git clone https://github.com/Brickzhuantou/CalibDepth
```
3. 安装依赖
```sh
pip install requirement.txt
```

### 文件目录说明

```
filetree 

├── /dataset/          ----数据读取
├── /environment/      ----标定执行相关函数
├── /ip_basic/         ----深度补全相关
├── /ipcv_utils/       ----可视化函数
├── /models/           ----模型搭建
├── /test/             ----函数功能测试
├── /utility/          ----通用函数
├── train.py           ----训练脚本
├── test.py            ----评测脚本
├── visual_test.py     ----可视化脚本
├── generate_depth_gt.py     ----深度图标签生成脚本
└── README.md
```

### 数据准备
1. Kitti官网下载数据 https://www.cvlibs.net/datasets/kitti/raw_data.php?type=road
  （也可以参考[CalibNet](https://github.com/epiception/CalibNet/tree/main)下载）

2. 数据集组织和命名如下；
```
├── /dataset/
    |── /kitti_raw
        |── /2011_09_26/
            |── /2011_09_26_drive_0001_sync/
                |── /depth_gt/
                |── /image_00/
                |── /image_01/
                |── /image_02/
                |── /image_03/
                |── /oxts/
                |── /velodyne_points/
            |── /2011_09_26_drive_0002_sync/
            |── /2011_09_26_drive_0005_sync/
            |── /2011_09_26_drive_0009_sync/
            |── ...
            |── calib_cam_to_cam.txt
            |── calib_imu_to_velo.txt
            |── calib_velo_to_cam.txt

        |── /2011_09_28/
            |── ...
        |── /2011_09_29/
            |── ...
        |── /2011_09_30/
            |── ...
        |── /2011_10_03/
            |── ...
    |── train.txt                    ----训练数据路径
    |── test.txt                     ----测试数据路径

```
说明：
    /depth_gt/存储用于单目深度估计的深度图标签，原始KITTI数据没有提供，可以参考generate_depth_gt.py脚本生成；
    train.txt和test.txt为随机采样生成的训练数据与测试数据路径；





### 运行
    修改train.py中的参数，执行 python train.py 即可；
    test.py同理；


### 鸣谢


- [CalibNet](https://github.com/epiception/CalibNet/tree/main)
- [LCCNet](https://github.com/IIPCVLAB/LCCNet)
- [reagent](https://github.com/dornik/reagent)
- [ip_basic](https://github.com/kujason/ip_basic)
- [Best_README_template](https://github.com/shaojintian/Best_README_template)


<!-- links -->
[your-project-path]:https://github.com/Brickzhuantou/CalibDepth

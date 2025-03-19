import argparse
import os
import time
import logging
import random

import numpy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_cifar
import numpy as np
from torch.autograd import Variable
from utils import *
from modules import *
from datetime import datetime
import dataset
from torch.utils.tensorboard import SummaryWriter


# 滞迟比较  base: 87.57

# base
# baseline  normal block ste： w：1  a：san1-2
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0')  # 87.86
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_2')  # 87.86
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0')  # 87.7
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_dzh_epoch-1-1_*fn_*w_1.0_dzh_2.2_4.2')  # 87.14
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_dzh_epoch-1-1_*fn_*w_1.0_dzh_2.2')  # 87.78

# fp-w
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_fp-w')  # 89.48
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_fp-w-2')  # 89.44
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_fp-w')  # 88.69
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_fp-w')  # 86.49

# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_fp-w-retrain')  # 第二阶段  88.39
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_fp-w-retrain')  # 第二阶段  88.53


# *fn *w 1.0 dzh 2.2  base: resnet20_nb_wd_5e-4_0_fp-w  89.48
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_fp-w-retrain-dzh1')  # 第二阶段  88.61
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-3_0_fp-w-retrain-dzh1')  # 第二阶段  88.53
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_fp-w-retrain-dzh1')  # 第二阶段  todo 88.71
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_fp-w-retrain-dzh1-k')  # 第二阶段  88.58
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_fp-w-retrain-dzh1')  # 第二阶段  88.2
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_2e-4_0_fp-w-retrain-dzh1')  # 第二阶段  88.23
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_fp-w-retrain-dzh1-seed-13')  # 换种子无效，权值已经初始化完毕
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_fp-w-retrain-dzh1-w-san')  # 第二阶段  88.25
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-5_0_fp-w-retrain-dzh1')  # 第二阶段  88.5
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-5_0_fp-w-retrain-dzh1')  # 第二阶段  88.17

# 补充实验  消融 (nb) ieee
# 1、baseline
# writer = SummaryWriter('./runs_ieee/resnet20_nb_wd_5e-4_0')  # 87.86
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0')  # 87.7 (87.3)

# 2、bs + wh
# writer = SummaryWriter('./runs_ieee/resnet20_nb_wd_5e-4_0_dzh0')  # 88.18
# writer = SummaryWriter('./runs_ieee/resnet20_nb_wd_1e-4_0_dzh0')  # 87.75

# 3、bs + wh + xz修正
# writer = SummaryWriter('./runs_ieee/resnet20_nb_wd_5e-4_0_dzh1')  # 88.36
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_dzh_epoch-1-1_*fn_*w_1.0_dzh_2.2')  # 87.78

# 4、bs + wh + xz + tsq
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_fp-w')  # 89.48
# writer = SummaryWriter('./runs_ieee/resnet20_nb_wd_5e-4_0_retrain')  # 88.39
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_retrain_dzh1')  # 88.61
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_1e-4_0_retrain_dzh1')  # 88.71  （3080：88.8）

# 补充实验  消融 (ieblock) ieee
# 1、baseline
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4')  # 87.57

# 2、bs + wh
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_dzh0')  # 87.69

# 3、bs + wh + xz
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_dzh1')  # 88.0

# 改 wd
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_1e-3_dzh1')  # 87.34
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_1e-3_5e-4_dzh1')  # 87.3
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_1e-3_dzh1')  # 86.83
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_1e-4_dzh1')  # 87.16
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_5e-5_dzh1')  # 86.37
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-3_dzh1')  # 不行

# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_1e-4_dzh1')  # 86.44
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-5_dzh1')  # 83.55

# 改 dzh1

# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_dzh1_lam_0.5')  # 87.94
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_dzh1_lam_2.0')  # 87.19

# 改 ieblock ste
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_0')  # 88.04

# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_0_dzh1')  # 88.08
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_1e-4_0_dzh1')  # 87.48
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_1e-3_0_dzh1')  # 87.98
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-5_0_dzh1')  # 86.71

# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_2e-4_dzh1')  # 87.94
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-4_dzh1')  # 88.08
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_5e-5_dzh1')  # 88.23
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1')  # 88.39
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_5e-6_dzh1')  # 87.9

# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_1e-3_5e-4_dzh1')  # 87.5
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_1e-3_1e-4_dzh1')  # 88.08
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_1e-3_5e-5_dzh1')  # 88.3

# 4、bs + wh + xz + tsq
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_fp-w')  # 89.75
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_retrain_dzh0')  #  87.66
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_retrain_dzh1')  #  87.84
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-5_retrain_dzh1')  #  88.15

# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_fp-w')  # 90.16
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain')  # 88.26
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain_dzh1')  # 88.77
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain_5e-4_0_dzh1')  # 88.4

# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_0_fp-w')  # 89.84
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_0_retrain_dzh1')  # 88.26
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_0_retrain_5e-4_1e-5_dzh1')  # 88.45
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_0_retrain')  # 88.41

# 消融结果：
# FWBA
# writer = SummaryWriter('./runs_ieee/resnet20_ie_wd_5e-4_fp-w')  # 89.75
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_0_fp-w')  # 89.84
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_fp-w')  # 90.16
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_fp-w')  # 89.58 保存模型

# Baseline
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4')  # 87.17
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5')  # 87.99
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_seed_33')  # 88.14

# Baseline* : retrain FWBA
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_retrain')  # 87.52
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain')  # 88.26
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain_seed_13')  # 88.61
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain_seed_22')  # 88.43
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain_seed_33')  # 88.26
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain_seed_44')  # 88.43
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain_seed_88')  # 88.24

# Baseline + FS
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh0')  # 88.23
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_dzh0')  # 87.6

# Baseline + FS + SC
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1')  # 88.39

# Baseline* + FS + SC  retrain FWBA
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_retrain_dzh1')  # 87.71
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_retrain_dzh1')  # 88.77

# -------------------------------------------------------------------------------------------------------
# 调参 lamda
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_0.1')   # 88.12
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_0.2')   # 87.91
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_0.4')   # 88.4
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_0.6')   # 88.56
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_0.8')   # 87.93
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1')   # 88.39
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_1.2')   # 88.11

# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_2.0')   # 87.97
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_4.0')   # 86.8
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_6.0')   # 86.06
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_8.0')   # 85.3
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_10.0')   # 84.23
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_lam_20.0')   # 79.94

# 调参 alpha 包含lr
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_alp_1.0')   # 87.93
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_alp_10.0')   # 87.83
# writer = SummaryWriter('./runs_ieee/resnet20_ie_san_wd_5e-4_1e-5_dzh1_alp_100.0')   # 86.79

# 调参 theta

# 分阶段训练 base  5e-5  (数据统一)
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5')  # 87.0   base
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_fp_w')  # 88.54
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_fp_a')  # 89.82
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_fp')  # 90.14   (92.1)

# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_fp_w_retrain')  # 87.69
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_fp_a_retrain')  # 85.92

# dzh retrain
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_fp_a_retrain-dzh1')  # 86.6
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_fp_w_retrain-dzh1')  # 87.62 低于直接二值retrain


# 改w为san
# base: resnet20_nb_wd_5e-4_0_fp-w
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4_0_fp-w-retrain-dzh1-w-san')  # 88.44

# 5.1、5.2
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_dzh_epoch-1-1_*fn_*w_1.0_dzh_2.2_5.2')  # 87.81
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_dzh_epoch-1-1_*fn_*w_1.0_dzh_2.2_5.2_2')  # 88.38
# writer = SummaryWriter('./runs_wus/resnet20_nb_wd_5e-4_0_dzh_epoch-1-1_*fn_*w_1.0_dzh_2.2_5.1')  # 88.22

## 大论文实验
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4_0_2')  # 87.86
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4')  # 83.88


# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_1e-5')  # 85.6
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_1e-4')  # 87.34
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_1e-3')  # 80.88


# -------------------------完整 dzh1 (每个epoch开始，进行符号修正)---------------------------
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4_0_dzh1_*c_0.1')  # 65.35
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4_0_dzh1_*c_0.01')  # 72.32
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4_0_dzh1_*c_0.001')  # 87.55
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4_0_dzh1_*c_0.0001')  # 88.46 (已存)

# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4_0_dzh1_*lr_1.0')  # 64.4
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-4_0_dzh1_*lr_0.1')  # 77.54

# -------------------------------------------------------------------------------------

# 动态阈值 补充 alpha
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh2_*w_1.0_ttt')  # 87.32 可以复现

# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh2_*w_*alpha_100.0')  # 86.3
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh2_*w_*alpha_10.0')  # 86.9
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh2_*w_*alpha_1.0')  # 87.4 (87.1)
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh2_*w_*alpha_0.1')  # 87.1 (86.6)

# 最终策略
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0')  # 87.2

# 带符号修正的 更新频率 实验  dzh1
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_50')  # 86.79
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_100')  # 87.23
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_195')  # 86.69
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_200')  # 86.87
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_300')  # 87.08
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_400')  # 87.23
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_500')  # 86.94
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_600')  # 87.19
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_800')  # 86.89
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_1000')  # top 87.35
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_2000')  # 87.11
# writer = SummaryWriter('./runs_wus/f-resnet20_nb_wd_5e-5_dzh1_*fn_*w_1.0_f_4000')  # 86.91

def statistic(model, total_iter):
    index = 0
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            x = m.weight.data[1, 1, 1, 1]
            y = m.weight.grad.data[1, 1, 1, 1]
            writer.add_scalar('weight/{}'.format(index), x, total_iter)
            writer.add_scalar('weight_grad/{}'.format(index), y, total_iter)
            index += 1


class BinOp():
    def __init__(self, model):
        # count the number of Conv2d
        global w_need_stop_wd, w_need_de_wd
        global g_zh_po, g_zh_ne, g_isZh_po, g_isZh_ne, g_isZh, g_zh_th, de_wd

        self.model = model

        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, BinarizeConv2d):
                count_Conv2d = count_Conv2d + 1

        self.num_of_params = count_Conv2d
        self.saved_params = []
        self.target_modules = []
        self.last_saved_params = []
        # 统计权值翻转情况
        self.w_flip_record = []
        self.w_flip_record_period = []
        self.reset_w_need_stop_wd = []
        self.init_zh = []
        self.w_border_po = []
        self.w_border_ne = []
        self.w_flip_record_period2 = []

        # 抑制权值更新
        self.w_begin_status_record = []
        self.tth = []
        # self.current_grad = []

        self.clip_th = 0
        self.total_nums = 0

        # 冻结权值
        self.freeze = []

        # flip by layer
        self.flip_by_layer = []
        self.flip_by_layer_zh = []

        self.grad_normal_ema = []
        self.flip_count = 0
        self.flip_count_zh = 0
        self.current_ep = 0
        for m in model.modules():
            if isinstance(m, BinarizeConv2d):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)          # 上一个epoch的weight
                self.last_saved_params.append(tmp)
                self.w_flip_record.append(torch.zeros_like(tmp))
                self.w_flip_record_period.append(torch.zeros_like(tmp))
                self.w_flip_record_period2.append(torch.zeros_like(tmp))
                self.w_begin_status_record.append(torch.zeros_like(tmp))
                self.tth.append(torch.zeros_like(tmp))
                self.reset_w_need_stop_wd.append(torch.ones_like(tmp))
                w_need_de_wd.append(torch.zeros_like(tmp))
                de_wd.append(torch.zeros_like(tmp))
                self.init_zh.append(torch.zeros_like(tmp))
                self.target_modules.append(m.weight)   # 当前epoch更新后的 weight
                self.grad_normal_ema.append(0)
                g_isZh.append(torch.zeros_like(tmp))
                g_zh_th.append(torch.zeros_like(tmp))
                self.w_border_po.append(torch.zeros_like(tmp))
                self.w_border_ne.append(torch.zeros_like(tmp))
                self.freeze.append(torch.zeros_like(tmp) == 1)
                # self.current_grad.append(torch.zeros_like(tmp))
                self.flip_by_layer.append(torch.zeros_like(tmp))
                self.flip_by_layer_zh.append(torch.zeros_like(tmp))

        self.count_time = 197
        self.count_time_epoch = 5
        self.maintain_sign_count = 0
        # self.w_scale = 1 + 1e-1
        self.w_scale_up = 1 + 1e-1
        self.w_scale_low = 1 - 1e-1

        w_need_stop_wd = self.reset_w_need_stop_wd

        # self.freeze = self.freeze == 1

        g_zh_ne = self.init_zh
        g_zh_po = self.init_zh
        g_isZh_po = self.init_zh
        g_isZh_ne = self.init_zh

        self.unflip_rate = []

        self.update_ep = 0

        self.update_init = True

        for i in range(self.num_of_params):
            self.unflip_rate.append(1.0)

    def save_params(self, train_iter):
        # 每个iter都执行
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def calculate_flip(self, ep, train_iter, cur_lr):
        global w_need_stop_wd,w_need_de_wd
        global g_zh_po, g_zh_ne, g_isZh_po, g_isZh_ne, g_isZh, g_zh_th,de_wd

        if self.current_ep != ep:
            # writer.add_scalar('flip_count', self.flip_count / 195, ep - 1)
            writer.add_scalar('flip_count', self.flip_count, ep - 1)
            writer.add_scalar('flip_count_zh', self.flip_count_zh, ep - 1)
            # flip by layer
            for index in range(self.num_of_params):
                x = self.flip_by_layer[index]
                num = x.sum()
                avg = num / x.numel()
                max_num = torch.max(x)
                rate = (x > 0).sum()
                rate = rate / x.numel()
                writer.add_scalar('layer_total/{}'.format(index), num, ep)
                writer.add_scalar('layer_avg/{}'.format(index), avg, ep)
                writer.add_scalar('layer_max/{}'.format(index), max_num, ep)
                writer.add_scalar('layer_rate/{}'.format(index), rate, ep)

                y = self.flip_by_layer_zh[index]
                num = y.sum()
                avg = num / y.numel()
                max_num = torch.max(y)
                rate = (y > 0).sum()
                rate = rate / y.numel()
                writer.add_scalar('layer_total_zh/{}'.format(index), num, ep)
                writer.add_scalar('layer_avg_zh/{}'.format(index), avg, ep)
                writer.add_scalar('layer_max_zh/{}'.format(index), max_num, ep)
                writer.add_scalar('layer_rate_zh/{}'.format(index), rate, ep)

            self.flip_count = 0
            self.flip_count_zh = 0
            for index in range(self.num_of_params):

                beta = cur_lr * 1.0                      # *lr

                # beta = 1.0
                self.w_begin_status_record[index] = torch.sign(self.saved_params[index])
                self.tth[index] = torch.abs(self.saved_params[index]) * beta       # stth
                # self.tth[index] = torch.ones_like(self.saved_params[index]) * beta   # ctth

                cnt = (self.w_flip_record_period[index] > 0).sum()
                self.unflip_rate[index] = 1 - (cnt / self.w_flip_record_period[index].numel())
                x = self.unflip_rate[index]
                writer.add_scalar('unflip_rate/{}'.format(index), x, ep - 1)
                y = self.w_flip_record_period[index]
                writer.add_histogram('flip_record/{}'.format(index), y, ep - 1)
                if ep - self.update_ep == 2:
                    self.w_flip_record_period2[index] = torch.zeros_like(self.w_flip_record_period2[index])
                    # beta = cur_lr * 1.0
                    # self.w_begin_status_record[index] = torch.sign(self.saved_params[index])
                    # self.tth[index] = torch.abs(self.saved_params[index]) * beta
                self.w_flip_record_period[index] = torch.zeros_like(self.w_flip_record_period[index])
            if ep - self.update_ep == 2:
                self.update_ep = ep

        self.total_nums = 0
        layer3_index = [17, 16, 15, 14, 13, 12]
        layer2_index = [11, 10, 9, 8, 7, 6]
        layer1_index = [5, 4, 3, 2, 1, 0]

        for index in range(self.num_of_params):

            w_sign_change = self.saved_params[index].sign() != self.target_modules[index].data.sign()
            self.w_flip_record[index][w_sign_change] += 1
            self.w_flip_record_period[index][w_sign_change] += 1
            self.w_flip_record_period2[index][w_sign_change] += 1

            self.flip_by_layer[index][w_sign_change] += 1

            tmp_0 = torch.zeros_like(self.saved_params[index])
            self.total_nums += (self.w_flip_record[index] == tmp_0).sum()
            self.flip_count += (self.saved_params[index].sign() != self.target_modules[index].data.sign()).sum()

        ind = 0
        for m in self.model.modules():
            if isinstance(m, BinarizeConv2d):
                self.flip_by_layer_zh[ind] += m.flip_record_zh
                self.flip_count_zh += m.flip_record_zh.sum()
                ind += 1

        if self.current_ep != ep:
            self.current_ep = ep

        writer.add_scalar('w_unflip_nums', self.total_nums, train_iter)


iter_count_train = 0
iter_count_val = 0
customLoss = None
w_need_stop_wd = []
w_need_de_wd = []
# zh
g_zh_po = []
g_zh_ne = []
g_isZh_po = []
g_isZh_ne = []
g_isZh = []
g_zh_th = []

# 反衰减
de_wd = []

conv_modules = []

def main():
    global args, best_prec1, conv_modules
    # global model
    global customLoss
    global w_need_stop_wd
    # 断点恢复
    # args.resume = True

    best_prec1 = 0
    if args.evaluate:
        args.results_dir = '/tmp'
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not args.resume and not args.evaluate:
        with open(os.path.join(save_path, 'config.txt'), 'w') as args_file:
            args_file.write(str(datetime.now()) + '\n\n')
            for args_n, args_v in args.__dict__.items():
                args_v = '' if not args_v and not isinstance(args_v, int) else args_v
                args_file.write(str(args_n) + ':  ' + str(args_v) + '\n')

        setup_logging(os.path.join(save_path, 'logger.log'))
        logging.info("saving to %s", save_path)
        logging.debug("run arguments: %s", args)
    else:
        setup_logging(os.path.join(save_path, 'logger.log'), filemode='a')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        if args.seed > 0:
            set_seed(args.seed)
        else:
            cudnn.benchmark = True
    else:
        args.gpus = None

    if args.dataset == 'tinyimagenet':
        num_classes = 200
        model_zoo = 'models_imagenet.'
    elif args.dataset == 'imagenet':
        num_classes = 1000
        model_zoo = 'models_imagenet.'
    elif args.dataset == 'cifar10':
        num_classes = 10
        model_zoo = 'models_cifar.'
    elif args.dataset == 'cifar100':
        num_classes = 100
        model_zoo = 'models_cifar.'

    # * create model
    if len(args.gpus) == 1:
        print('single gpu')
        model = eval(model_zoo + args.model)(num_classes=num_classes).cuda()
    else:
        print('multiple gpus')
        model = nn.DataParallel(eval(model_zoo + args.model)(num_classes=num_classes))
    if not args.resume:
        logging.info("creating model %s", args.model)
        logging.info("model structure: ")
        for name, module in model._modules.items():
            logging.info('\t' + str(name) + ': ' + str(module))
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

    bin_op = BinOp(model)

    # * evaluate
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            logging.error('invalid checkpoint: {}'.format(args.evaluate))
            return
        else:
            checkpoint = torch.load(args.evaluate)
            if len(args.gpus) > 1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         args.evaluate, checkpoint['epoch'])

    elif args.resume:
        checkpoint_file = os.path.join(save_path, 'checkpoint.pth.tar')
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            if len(args.gpus) > 1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    # 分阶段训练
    # base_path = './wus-model/'
    # model_path = 'resnet20-nb-5e-4-0-fp-w'  # 预训练模型 fp-w
    #
    # base_path = './kjzx-model/'
    # model_path = 'resnet20_base_fp-w_wd_5e-4_0'  # 预训练模型 fp-w

    # base_path = './wus-model/fi/'
    # model_path = 'f-resnet20_nb_wd_5e-5_fp_w-88.54'  # 预训练模型 fp-w
    # # model_path = 'f-resnet20_nb_wd_5e-5_fp_a-89.82'  # 预训练模型 fp-a

    # base_path = './wus-model/'
    # # model_path = 'resnet20_ie_wd_5e-4_fp-w-89.8'  # 预训练模型 fp-w 1
    # model_path = 'resnet20_ie_san_wd_5e-4_1e-5_fp-w-90.16'  # 预训练模型 fp-w 2
    # # model_path = 'resnet20_ie_san_wd_5e-4_0_fp-w-89.84'  # 预训练模型 fp-w 3
    # # model_path = 'resnet20_ie_san_wd_5e-4_fp-w-89.58'  # 预训练模型 fp-w 4
    # #
    # checkpoint_file = os.path.join(base_path + model_path, 'model_best.pth.tar')
    # checkpoint = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint['state_dict'])
    # # print('预训练模型加载完成')
    # logging.info('预训练模型加载完成')


    criterion = nn.CrossEntropyLoss().cuda()
    criterion = criterion.type(args.type)
    model = model.type(args.type)

    if args.evaluate:
        val_loader = dataset.load_data(
            type='val',
            dataset=args.dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            batch_size_test=args.batch_size_test,
            num_workers=args.workers)
        with torch.no_grad():
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0)
        logging.info('\n Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
        return

    # * load dataset
    train_loader, val_loader = dataset.load_data(
        dataset=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        num_workers=args.workers)

    all_params = model.parameters()
    # bin_params = []
    dada_params = []
    conv_params = []
    bconv_params = []
    ab_params = []

    for pname, p in model.named_parameters():
        # print(pname)
        if 'se' in pname:
            dada_params += [p]
        if 'conv' in pname or 'my.1' in pname or 'linear' in pname:
            # print(pname)
            conv_params += [p]
        if 'layer' in pname and ('conv1' in pname or 'conv2' in pname):
            bconv_params += [p]

    params_id = list(map(id, bconv_params))
    # ab_id = list(map(id, ab_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    # other_params = list(filter(lambda p: id(p) not in ab_id, other_params))

    # * optimizer settings
    if args.optimizer == 'sgd':
        # 修改wd
        optimizer = torch.optim.SGD(
            [
                # {'params': dada_params, 'initial_lr': args.lr, "weight_decay": 1e-3},
                # {'params': other_params, 'initial_lr': args.lr, "weight_decay": 1e-3}
                # {'params': conv_params, 'initial_lr': args.lr, "weight_decay": 5e-3},
                # {'params': other_params, 'initial_lr': args.lr, "weight_decay": 1e-4}
                {'params': bconv_params, 'initial_lr': args.lr, "weight_decay": 5e-4, 'lr': args.lr},
                {'params': other_params, 'initial_lr': args.lr, "weight_decay": 1e-5, 'lr': args.lr}
            ],
            # lr=args.lr,
            momentum=args.momentum
        )

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}],
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        logging.error("Optimizer '%s' not defined.", args.optimizer)

    if args.lr_type == 'cos':
        # 原始
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm_up * 4, eta_min=0,
                                                                  last_epoch=args.start_epoch)
        # 两阶段变化1：1
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (args.epochs-args.warm_up*4)//2, eta_min = 0, last_epoch=args.start_epoch)

    elif args.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1, last_epoch=-1)
    elif args.lr_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (
                    1.0 - (epoch - args.warm_up * 4) / (args.epochs - args.warm_up * 4)), last_epoch=-1)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        logging.info("criterion: %s", criterion)
        logging.info('scheduler: %s', lr_scheduler)

    # * record names of conv_modules
    conv_modules = []
    for name, module in model.named_modules():
        if isinstance(module, BinarizeConv2d):
            conv_modules.append(module)
            # print(name)

    customLoss = CustomLoss(model)

    for epoch in range(args.start_epoch + 1, args.epochs):
        time_start = datetime.now()
        # * warm up
        if args.warm_up and epoch < 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch + 1) / 5

        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])
            # break
        # logging.info('alpha lr: %s', optimizer.param_groups[0]['lr'])
        # logging.info('other lr: %s', optimizer.param_groups[1]['lr'])

        # * compute threshold tau
        # tau = cpt_tau(epoch)
        # print(optimizer.param_groups[0]['lr'])
        for module in conv_modules:
            # 分阶段量化a
            module.epoch = torch.tensor([epoch]).float()
            module.lr = optimizer.param_groups[0]['lr']
            # print(module.lr)

        # * training
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer, bin_op)

        # * adjust Lr
        if epoch >= 4 * args.warm_up:
            lr_scheduler.step()

        # * evaluating
        with torch.no_grad():
            val_loss, val_prec1, val_prec5 = validate(
                val_loader, model, criterion, epoch, bin_op, optimizer)

        # * remember best prec
        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = max(val_prec1, best_prec1)
            best_epoch = epoch
            best_loss = val_loss

        # * save model
        if epoch % 1 == 0:
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            model_optimizer = optimizer.state_dict()
            model_scheduler = lr_scheduler.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model_state_dict,
                'best_prec1': best_prec1,
                'optimizer': model_optimizer,
                'lr_scheduler': model_scheduler,
            }, is_best, path=save_path)

        if args.time_estimate > 0 and epoch % args.time_estimate == 0:
            time_end = datetime.now()
            cost_time, finish_time = get_time(time_end - time_start, epoch, args.epochs)
            logging.info('Time cost: ' + cost_time + '\t'
                                                     'Time of Finish: ' + finish_time)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        writer.add_scalar('train_prec1', train_prec1, epoch + 1)
        writer.add_scalar('val_prec1', val_prec1, epoch + 1)

    logging.info('*' * 50 + 'DONE' + '*' * 50)
    logging.info('\n Best_Epoch: {0}\t'
                 'Best_Prec1 {prec1:.4f} \t'
                 'Best_Loss {loss:.3f} \t'
                 .format(best_epoch + 1, prec1=best_prec1, loss=best_loss))


total_iter = 0


def forward(data_loader, model, criterion, epoch=0, bin_op=None, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global total_iter

    global iter_count_train, iter_count_val, conv_modules

    global customLoss

    global g_zh_po, g_zh_ne, g_isZh_po, g_isZh_ne, g_isZh, g_zh_th

    end = time.time()

    for i, (inputs, target) in enumerate(data_loader):
        # * measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        input_var = Variable(inputs.type(args.type))
        target_var = Variable(target)

        lr = optimizer.param_groups[0]['lr']
        for module in conv_modules:
            module.train_iter = iter_count_train

        # * compute output
        output = model(input_var)
        loss_ce = criterion(output, target_var)
        # loss_bwd, loss_owd, loss_w, loss_de_bwd = customLoss(iter_count_train, epoch)
        lam = 0.0001

        loss = loss_ce

        if training:
            # writer.add_scalar('train_ce_loss', loss_ce.data.item(), iter_count_train + 1)
            # writer.add_scalar('train_w_loss', loss_w.data.item(), iter_count_train + 1)
            iter_count_train = iter_count_train + 1
        else:
            # writer.add_scalar('val_ce_loss', loss_ce.data.item(), iter_count_val + 1)
            # writer.add_scalar('val_w_loss', loss_w.data.item(), iter_count_val + 1)
            iter_count_val = iter_count_val + 1

        if type(output) is list:
            output = output[0]

        # * measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            bin_op.save_params(iter_count_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bin_op.calculate_flip(epoch, iter_count_train, lr)
            statistic(model, total_iter)
            total_iter += 1

        # * measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(data_loader),
                phase='TRAINING' if training else 'EVALUATING',
                batch_time=batch_time,
                data_time=data_time, loss=losses,
                top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, bin_op):
    model.train()
    return forward(data_loader, model, criterion, epoch, bin_op,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch, bin_op, optimizer):
    model.eval()
    return forward(data_loader, model, criterion, epoch, bin_op,
                   training=False, optimizer=optimizer)


class CustomLoss(nn.Module):
    def __init__(self, model):
        super(CustomLoss, self).__init__()
        self.cur_bwd_loss = torch.tensor([0.0]).cuda()
        self.cur_owd_loss = torch.tensor([0.0]).cuda()
        self.cur_wd_loss = torch.tensor([0.0]).cuda()
        self.cur_de_loss = torch.tensor([0.0]).cuda()

        self.cur_mse_loss = torch.tensor([0.0]).cuda()
        self.update_times = 500
        self.update_epochs = 2
        self.first_iter_update = False
        self.last_epoch = -1
        self.cur_grad_sum = []
        self.pre_weight = []
        for m in model.modules():
            if isinstance(m, BinarizeConv2d):
                self.pre_weight.append(m.weight.data.clone())

        self.model = model
        self.epoch2alpha_list = cos_min2max_1(0.1, 0.0)

    def forward(self, current_iter, current_epoch):
        global w_need_stop_wd,w_need_de_wd,de_wd
        model = self.model
        self.cur_mse_loss = torch.tensor([0.0]).cuda()
        self.cur_bwd_loss = torch.tensor([0.0]).cuda()
        self.cur_owd_loss = torch.tensor([0.0]).cuda()
        self.cur_wd_loss = torch.tensor([0.0]).cuda()
        self.cur_de_loss = torch.tensor([0.0]).cuda()
        # 权值的loss
        index = 0
        for m in model.modules():
            if hasattr(m, 'weight'):
                tmp = m.weight
                self.cur_wd_loss += torch.sum(torch.pow(tmp, 2))
                layer_index = 1
                if isinstance(m, BinarizeConv2d):

                    if m.weight.size(0) == 16:
                        layer_index = 1
                    elif m.weight.size(0) == 32:
                        layer_index = 2
                    elif m.weight.size(0) == 64:
                        layer_index = 3

                    ## origin
                    # self.cur_bwd_loss += torch.sum(torch.pow(tmp, 2))

                    ## w_need_stop_wd
                    wd = w_need_stop_wd[index]
                    self.cur_bwd_loss += torch.sum(torch.pow(tmp, 2) * wd)
                    index = index + 1

                else:
                    self.cur_owd_loss += torch.sum(torch.pow(tmp, 2))

        return self.cur_bwd_loss, self.cur_owd_loss, self.cur_mse_loss, self.cur_de_loss


def cos_min2max_1(start, end):
    ret = []
    for i in range(600):
        start = (1 + torch.cos(torch.pi * i / torch.tensor(600.0))) / (
                1 + torch.cos(torch.pi * (i - 1) / torch.tensor(600.0))) * (start - end) + end
        ret.append(start.cuda())
    return ret


if __name__ == '__main__':
    main()

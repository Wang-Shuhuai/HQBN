import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable

from cifar.utils.options import args
# from utils.options import args

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, layer_index=0, *kargs, **kwargs):  # resnet20
    # def __init__(self, *kargs, **kwargs):    # resnet18
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        # epoch
        self.epoch = torch.tensor([0.0])
        self.lr = torch.tensor([0.0])
        # layer_index
        # self.layer_index = layer_index
        self.layer_index = 0

        # 滞回参数
        self.zh_po = torch.zeros_like(self.weight.data).cuda()  # 正 -> 负  延迟为 zh_po
        self.zh_ne = torch.zeros_like(self.weight.data).cuda()  # 负 -> 正  延迟为 zh_ne
        self.isZh_po = torch.zeros_like(self.weight.data).cuda()  # 需要为 1
        self.isZh_ne = torch.zeros_like(self.weight.data).cuda()  # 需要为 1
        self.isZh = torch.zeros_like(self.weight.data).cuda()  # 需要为 1, 否则为0
        self.zh_th = torch.zeros_like(self.weight.data).cuda()  # 需要为 1, 否则为0

        self.last_w = torch.zeros_like(self.weight.data).cuda()
        self.last_w_init = torch.zeros_like(self.weight.data).cuda()
        self.last_sign = torch.zeros_like(self.weight.data).cuda()
        self.last_ep = torch.tensor([0.0])
        self.zh_th_sign = torch.ones_like(self.weight.data).cuda()
        self.pre_iter_w = torch.zeros_like(self.weight.data).cuda()
        self.first_iter = True
        self.first_epoch = True
        self.train_iter = torch.tensor([0.0])
        self.w_flip_num = torch.zeros_like(self.weight.data).cuda()
        self.stop_zh = torch.ones_like(self.weight.data).cuda()
        self.lamda_scale = torch.ones_like(self.weight.data).cuda()
        self.flip_one = torch.zeros_like(self.weight.data).cuda()
        self.flip_record_zh = torch.zeros_like(self.weight.data).cuda()


    def forward(self, input):
        a = input
        w1 = self.weight

        ## 二值---------------------------------
        # bw = BinaryQuantize_1().apply(w1)
        # ba = BinaryQuantize_ste_2().apply(a)

        ## 双向滞回: base c / w--new--------------------------------
        lamda = 1.0
        alpha = 100.0
        # lamda = (1 - (self.epoch / 600.0)) * 0.001

        if self.first_iter:
            self.first_iter = False
            self.pre_iter_w = w1.data.clone()
            self.last_w = torch.abs(w1.data.clone())
            self.zh_th_sign = torch.sign(w1.data.clone())
            self.last_w_init = w1.data.clone()
            self.flip_record_zh = torch.zeros_like(self.flip_record_zh)
        else:
            if self.last_ep != self.epoch:
                # # xz修正 dzh2.2
                adjust_sign = torch.sign(w1.data + self.zh_th) != torch.sign(w1.data)
                w1.data[adjust_sign] = w1.data[adjust_sign] + self.zh_th[adjust_sign]
                #
                self.last_w = torch.abs(w1.data.clone())
                self.last_w_init = w1.data.clone()
                self.zh_th_sign = torch.sign(w1.data.clone())
                self.pre_iter_w = w1.data.clone()
                self.w_flip_num = torch.zeros_like(self.w_flip_num)
                #
                # self.zh_th = self.zh_th_sign * lamda    # c
                # self.zh_th = self.zh_th_sign * lamda * self.lr      # cos
                # self.zh_th = self.last_w * self.zh_th_sign * lamda      # w
                # self.zh_th = self.last_w * self.zh_th_sign * lamda * self.lr  # w * lr
                # self.zh_th = self.last_w * self.zh_th_sign * (self.w_flip_num + 1) * lamda * self.lr * 10.0

                self.zh_th = self.last_w * self.zh_th_sign * (self.w_flip_num + 1) * lamda  # dzh1
                # self.zh_th = self.last_w * self.zh_th_sign * (self.w_flip_num + 1) * lamda * self.lr * alpha  # alpha调参

                self.last_ep = self.epoch
                self.first_epoch = False

            self.flip_record_zh = torch.zeros_like(self.flip_record_zh)

            mask = torch.sign(self.pre_iter_w + self.zh_th) != torch.sign(w1.data + self.zh_th)
            self.flip_record_zh[mask] = 1
            self.zh_th_sign[mask] = -1 * self.zh_th_sign[mask]
            self.w_flip_num[mask] += 1
            self.pre_iter_w = w1.data.clone()

        # self.zh_th = self.zh_th_sign * lamda.cuda()   # c
        # self.zh_th = self.zh_th_sign * lamda * self.lr  # cos
        # self.zh_th = self.last_w * self.zh_th_sign * lamda  # w
        # self.zh_th = self.last_w * self.zh_th_sign * lamda * self.lr  # w * lr

        self.zh_th = self.last_w * self.zh_th_sign * (self.w_flip_num + 1) * lamda  # dzh1
        # self.zh_th = self.last_w * self.zh_th_sign * (self.w_flip_num + 1) * lamda * self.lr * alpha  # alpha调参

        bw = BinaryQuantize_1_zh().apply(w1, self.zh_th)
        # ba = BinaryQuantize_san().apply(a)   # nb
        ba = BinaryQuantize_ste_2().apply(a)   # ieblock


        ## 全精度--------------------------------
        # bw = w1
        # ba = a

        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


class BinaryQuantize_1(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class BinaryQuantize_1_zh(Function):
    @staticmethod
    def forward(ctx, input, zh_th):
        w = input + zh_th
        out = torch.sign(w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class BinaryQuantize_san(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out == 0] = -1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))   # san-1-2
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input

class BinaryQuantize_san_zh(Function):
    @staticmethod
    def forward(ctx, input, zh_th):
        w = input + zh_th
        ctx.save_for_backward(w)
        out = torch.sign(w)
        out[out == 0] = -1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))   # san-1-2
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None

class BinaryQuantize_ste(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[input < -1] = 0
        grad_input[input > 1] = 0

        return grad_input

class BinaryQuantize_ste_2(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[input < -2] = 0
        grad_input[input > 2] = 0

        return grad_input

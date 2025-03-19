import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from utils.options import args

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

        # epoch
        self.epoch = torch.tensor([0.0])
        self.lr = torch.tensor([0.0])
        self.zh_th = torch.zeros_like(self.weight.data).cuda()
        self.last_w = torch.zeros_like(self.weight.data).cuda()
        self.last_w_init = torch.zeros_like(self.weight.data).cuda()
        self.last_ep = torch.tensor([0.0])
        self.zh_th_sign = torch.ones_like(self.weight.data).cuda()
        self.pre_iter_w = torch.zeros_like(self.weight.data).cuda()
        self.first_iter = True
        self.first_epoch = True
        self.train_iter = torch.tensor([0.0])
        self.w_flip_num = torch.zeros_like(self.weight.data).cuda()
        self.flip_one = torch.zeros_like(self.weight.data).cuda()
        self.flip_record_zh = torch.zeros_like(self.weight.data).cuda()

    def forward(self, input):
        a = input
        w1 = self.weight

        # WH --------------------------------
        lamda = 1.0
        if self.first_iter:
            self.first_iter = False
            self.pre_iter_w = w1.data.clone()
            self.last_w = torch.abs(w1.data.clone())
            self.zh_th_sign = torch.sign(w1.data.clone())
            self.last_w_init = w1.data.clone()
            self.flip_record_zh = torch.zeros_like(self.flip_record_zh)
        else:
            if self.last_ep != self.epoch:
                # # 修正 dzh2.2
                adjust_sign = torch.sign(w1.data + self.zh_th) != torch.sign(w1.data)
                w1.data[adjust_sign] = w1.data[adjust_sign] + self.zh_th[adjust_sign]

                self.last_w = torch.abs(w1.data.clone())
                self.last_w_init = w1.data.clone()
                self.zh_th_sign = torch.sign(w1.data.clone())
                self.pre_iter_w = w1.data.clone()
                self.w_flip_num = torch.zeros_like(self.w_flip_num)

                self.zh_th = self.last_w * self.zh_th_sign * (self.w_flip_num + 1) * lamda

                self.last_ep = self.epoch
                self.first_epoch = False

            self.flip_record_zh = torch.zeros_like(self.flip_record_zh)

            mask = torch.sign(self.pre_iter_w + self.zh_th) != torch.sign(w1.data + self.zh_th)
            self.flip_record_zh[mask] = 1
            self.zh_th_sign[mask] = -1 * self.zh_th_sign[mask]
            self.w_flip_num[mask] += 1
            self.pre_iter_w = w1.data.clone()

        self.zh_th = self.last_w * self.zh_th_sign * (self.w_flip_num + 1) * lamda

        bw = BinaryQuantize_1_zh().apply(w1, self.zh_th)
        # ba = BinaryQuantize_a().apply(a)
        ba = BinaryQuantize_a_ste.apply(a)

        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output



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


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input



class BinaryQuantize_a_ste(Function):
    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):

        # 原方法----------------------------
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        grad_input[input < -2] = 0
        grad_input[input > 2] = 0

        return grad_input

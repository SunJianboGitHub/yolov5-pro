#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   common.py
@Time    :   2022/09/26 16:47:18
@Author  :   YOLOv5 by JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   Common modules
'''

import math
import torch
import warnings
import torch.nn as nn


def autopad(kernel_size, padding=None, dilation=1):
    """
        kernel_size:    卷积核尺寸
        padding:        边缘填充的像素个数
        dilation:       膨胀卷积的空洞率
    """
    # Pad to 'same' shape outputs
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [dilation * (x - 1) + 1 for x in kernel_size]  # actual kernel-size
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]       # 自动边界填充
    return padding



class Conv(nn.Module):
    """
        标准卷积: nn.Conv2d + nn.BatchNorm2d + nn.SiLU
        激活函数: 可选择默认SiLU, 也可以自己实现, 还可以不使用激活函数
        合并推理: 将卷积层与BN层合并计算
    """
    
    def default_act(self):
        # default_act = nn.SiLU()                                     # 默认激活函数
        default_act = nn.LeakyReLU(0.1, inplace=True)
        return default_act

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1, act=True):
        super().__init__()
        default_act = nn.LeakyReLU(0.1, inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding, dilation), dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self.default_act() if act else act if isinstance(act, nn.Module) else nn.Identity()
            
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



class DWConv(Conv):
    """
        Depth-wise卷积: 一个卷积核负责一个通道, 一个通道只被一个卷积核卷积, 它与Point-wise合称为深度可分离卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, act=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, dilation=dilation, 
                        groups=math.gcd(in_channels, out_channels), act=act)



class Bottleneck(nn.Module):
    """
        标准瓶颈层: CBL_1x1 + CBL_3x3 + shortcut
        c_in:           输入通道数
        c_out:          输出通道数
        shortcut:       是否跳跃链接
        groups:         分组卷积的组数
        expansion:      通道数的扩展倍数
    """
    def __init__(self, c_in, c_out, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(c_out * expansion)
        self.cv1 = Conv(c_in, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, c_out, 3, 1, groups=groups)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # y1=CBL_1X1 + N*Bottleneck + nn.Conv2d
    # y2=nn.Conv2d
    # cat(y1, y2) + BN + act 
    # CBL_1x1
    """
        c_in:           输入通道数
        c_out:          输出通道数
        repeat:         Bottleneck重复的次数
        shortcut:       是否跳跃链接
        groups:         分组卷积的组数
        expansion:      隐层的通道扩展倍数
    """
    def __init__(self, c_in, c_out, repeat=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(c_out * expansion)
        self.cv1 = Conv(c_in, hidden_channels, 1, 1)
        self.cv2 = nn.Conv2d(c_in, hidden_channels, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.cv4 = Conv(2*hidden_channels, c_out, 1, 1)
        self.bn  = nn.BatchNorm2d(2*hidden_channels)                # 应用到cat(cv2, cv3)
        # self.act = nn.SiLU()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 1) for _ in range(repeat)])
    
    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottloneck with 3 convolutions
    # y1 = CBL_1x1 + N*Bottleneck
    # y2 = CBL_1x1
    # return cat(y1, y2) + CBL_1x1
    """
        c_in:           输入通道数
        c_out:          输出通道数
        repeat:         Bottleneck的重复次数
        shortcut:       是否跳跃链接
        groups:         分组卷积的组数
        expansion:      通道的扩展倍数
    """
    def __init__(self, c_in, c_out, repeat=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(c_out * expansion)
        self.cv1 = Conv(c_in, hidden_channels, 1, 1)
        self.cv2 = Conv(c_in, hidden_channels, 1, 1)
        self.cv3 = Conv(2*hidden_channels, c_out, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 1) for _ in range(repeat)])
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))




class BottleneckPruned(nn.Module):
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, groups=1):
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 3, 1)
        self.add = shortcut and cv1in == cv2out
        
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Pruned(nn.Module):
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, repeat=1, shortcut=True, groups=1):
        super(C3Pruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in+cv2out, cv3out, 1, 1)
        self.m = nn.Sequential(*[BottleneckPruned(*bottle_args[k], shortcut, groups) for k in range(repeat)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPFPruned(nn.Module):
    def __init__(self, cv1in, cv1out, cv2out, k=5):
        super(SPPFPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out * 4, cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                 # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


class SPP(nn.Module):
    #  Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    # x = CBL_1x1
    # y = maxpool 5x5 9x9 13x13 identity
    # cat(y) + CBL_1x1
    """
        c_in:               输入通道
        c_out:              输出通道
        kernel_size:        最大池化的尺寸, 默认是 5 9 13
    """
    def __init__(self, c_in, c_out, kernel_size=(5, 9, 13)):
        super().__init__()
        hidden_channels = c_out // 2
        self.cv1 = Conv(c_in, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels*(len(kernel_size)+1), c_out, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2) for k in kernel_size])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                 # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    """
        c_in:               输入通道数
        c_out:              输出通道数
        kernel_size:        最大池化的尺寸, 替代之前的(5, 9, 13)
    """
    def __init__(self, c_in, c_out, kernel_size=5):
        super().__init__()
        hidden_channels = c_out // 2
        self.cv1 = Conv(c_in, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels*4, c_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                 # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))



class Focus(nn.Module):
    # Focus wh information into c-space
    """
        c_in:               输入通道数
        c_out:              输出通道数
        kernel_size:        卷积核的尺寸
        stride:             卷积的步长
        padding:            边界填充
        groups:             分组卷积的组数
        act:                是否激活
    """
    def __init__(self, c_in, c_out, kernel_size=1, stride=1, padding=0, groups=1, act=True):
        super().__init__()
        self.cv = Conv(c_in*4, c_out, kernel_size, stride, padding, groups=groups, act=act)

    def forward(self, x):
        y = torch.cat((
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2]
        ), dim=1)
        return self.cv(y)
    
    def forward_export(self, x):
        return self.cv(x)
    


class FFocus(nn.Module):
    # Fast Focus with groups conv
    """
        c_in:               输入通道数
        c_out:              输出通道数
        down_scale:         下采样尺度, 必须是偶数 2 4 8
    """
    # 卷积核的尺寸=下采样倍数+1
    # 切片之后, 采用深度可分离卷积
    def __init__(self, c_in, c_out, down_scale=2):
        super().__init__()
        self.down_scale = down_scale
        expansion_channels = c_in * (self.down_scale ** 2)

        self.dw_conv = Conv(expansion_channels, expansion_channels, kernel_size=self.down_scale+1, stride=1, groups=expansion_channels)          # 这里采用DW卷积，直接修改groups=expansion_channels
        self.pw_conv = Conv(expansion_channels, c_out, 1, 1)                                                                                     # 这里采用PW卷积

    def forward(self, x):
        slice_list = []
        for i in range(self.down_scale):                            # 遍历行
            for j in range(self.down_scale):                        # 遍历列
                slice_list.append(x[..., j::self.down_scale, i::self.down_scale])               # 排布采用先行后列
        return self.pw_conv(self.dw_conv(torch.cat(slice_list, dim=1)))



class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    """
        dimension: 合并的维度
        x:          输入是一个网络层列表
    """
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, dim=self.dimension)



if __name__ == '__main__':
    layer = FFocus(3, 64, 4)
    a = torch.ones((1, 3, 640, 640), dtype=torch.float32)
    b = layer(a)
    print(b.shape)





















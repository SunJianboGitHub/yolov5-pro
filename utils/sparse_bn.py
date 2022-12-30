#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sparse_bn.py
@Time    :   2022/12/12 21:32:44
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   通道剪枝时使用,对某些BN权重执行L1正则化,带有跳跃链接的一般不剪枝
'''

import torch
import torch.nn as nn

from models.common import Bottleneck



class OptimizerBN:
    def __init__(self, model, sr=0.001, epochs=100):
        self.model = model                                      # 网络模型
        self.sr = sr                                            # 稀疏化系数,也就是L1正则化的系数
        self.epochs = epochs                                    # 用于动态改变权重系数
        self.ignore_bn_list = self.acquire_ignore_bn_list()     # 不需要进行稀疏训练的层,通常是bottleneck层
        
        
    def update_bn(self, epoch):
        # sr被称为稀疏率(sparsity rate),我觉得就是L1正则化系数
        srtmp = self.sr * (1 - 0.9 * epoch / self.epochs)                         # BN.weight的系数
        
        for k, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and (k not in self.ignore_bn_list):
                # 为什么对BN参数使用L1正则？因为它可以稀疏化参数,使参数趋于零
                # L1损失函数是绝对值,它的倒数是-1、0、1
                # 因此每一个参数加入L1正则之后的梯度是在原有梯度基础上加上一个数值(sr * sign(BN.weight.data))
                m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))                      # L1 正则,直接处理梯度      
                m.bias.grad.data.add_(self.sr * 10 * torch.sign(m.bias.data))                        # L1 正则,这个可以不需要


    def acquire_ignore_bn_list(self):
        ignore_bn_list = []
        for k, m in self.model.named_modules():
            if isinstance(m, Bottleneck):
                # 存在瓶颈层的这一块是不剪枝的
                # 因为剪枝之后不能保证输入输出通道数一致
                # 这里是针对C3层的处理策略,其它网络类似
                # 这里所说的C3是带跳跃链接的C3
                if m.add:                                                           # 存在跳跃链接
                    ignore_bn_list.append(k.rsplit(".", 2)[0] + "cv1.bn")              # C3网络层中的cv1卷积的BN不正则化
                    ignore_bn_list.append(k + ".cv1.bn")                            # BottleNect中的cv1卷积的BN
                    ignore_bn_list.append(k + ".cv2.bn")                            # BottleNect中的cv2卷积的BN
            # if isinstance(m, nn.BatchNorm2d) and (k not in self.ignore_bn_list):
                
            #     # 为什么对BN参数使用L1正则？因为它可以稀疏化参数,使参数趋于零
            #     # L1损失函数是绝对值,它的倒数是-1、0、1
            #     # 因此每一个参数加入L1正则之后的梯度是在原有梯度基础上加上一个数值(sr * sign(BN.weight.data))
            #     m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))                      # L1 正则,直接处理梯度      
            #     m.bias.grad.data.add_(self.sr * 10 * torch.sign(m.bias.data))                        # L1 正则,这个可以不需要
        return ignore_bn_list

    def display_bn(self):
        module_weight_list = []                                                                 # 存储每一个BN的权重值,也就是gamma的值
        module_bias_list = []                                                                   # 存储每一个BN的偏置的值,也就是beta
        for k, m in self.model.named_modules():                                                 # 遍历每一个网络层
            if isinstance(m, nn.BatchNorm2d) and (k not in self.ignore_bn_list):                # 如果当前层需要剪枝
                bn_w = m.state_dict()["weight"]                                                 # 获取BN层权重的值
                bn_b = m.state_dict()["bias"]                                                   # 获取BN层的偏置的值
                module_weight_list.append(bn_w)                                                 # 将每一层的BN权重添加到列表
                module_bias_list.append(bn_b)                                                   # 将每一层的BN偏置添加到列表
                
        size_list = [item.data.shape[0] for item in module_weight_list]                         # 确定每一层的BN权重的维度

        bn_weights = torch.zeros(sum(size_list))                                                # 创建一个大的tensor,维度是所有BN权重维度的展开
        bn_bias    = torch.zeros(sum(size_list))
        index = 0
        for i, size in enumerate(size_list):                                                    # 遍历所有的BN层列表
            bn_weights[index:(index + size)] = module_weight_list[i].data.abs().clone()         # 获取BN权重的值,并重新克隆赋值
            bn_bias[index:(index + size)]    = module_bias_list[i].data.abs().clone()           # 获取BN偏置的值,并重新克隆赋值
            index += size
        
        bn_weights = bn_weights.cpu()
        bn_bias = bn_bias.cpu()
        # bn_weights = bn_weights.cpu().tolist()
        # bn_bias = bn_bias.cpu().tolist()
        # bn_weights.sort()
        # bn_bias.sort()
        
        return bn_weights, bn_bias





























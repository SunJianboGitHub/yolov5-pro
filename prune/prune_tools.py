#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   prune_tools.py
@Time    :   2022/12/19 15:58:15
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   channel prune tools
'''


import torch


# 根据BN层，提取BN层的权重，根据稀疏比例值，找出BN阈值
def gather_bn_weights(model_bn_dict):
    size_list = [m.weight.data.shape[0] for m in model_bn_dict.values()]
    
    bn_weights = torch.zeros(sum(size_list))                                                # 创建一个大的tensor,维度是所有BN权重维度的展开
    bn_bias    = torch.zeros(sum(size_list))
    index = 0
    for i, m in enumerate(model_bn_dict.values()):                                          # 遍历所有的BN层列表
        size = size_list[i]
        bn_weights[index:(index + size)] = m.weight.data.abs().clone()         # 获取BN权重的值,并重新克隆赋值
        bn_bias[index:(index + size)]    = m.bias.data.abs().clone()           # 获取BN偏置的值,并重新克隆赋值
        index += size
    
    return bn_weights, bn_bias


# 根据BN阈值制作 BN mask
# 注意这里需要保证至少每一个BN层保留2个通道
def obtain_bn_mask(bn_module, threshold):
    mask = bn_module.weight.data.abs().ge(threshold).float()
    remain_chs = (mask == 1.0).sum()                                        # 统计裁剪后剩余的通道数量
    
    while remain_chs.item() <= 1:                                           # 这里为了保证剩余超过2个通道
        threshold -= 0.005
        mask = bn_module.weight.data.abs().ge(threshold).float()
        remain_chs = (mask == 1.0).sum()
        
    return mask























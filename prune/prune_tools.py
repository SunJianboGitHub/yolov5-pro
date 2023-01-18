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
import numpy as np
import torch.nn as nn
from datetime import datetime

from models.common import Bottleneck
from prune.sparse_tools import SparseBN



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
def obtain_bn_mask(bn_module, threshold, min_channel=2):
    mask = bn_module.weight.data.abs().ge(threshold).float()
    remain_chs = (mask == 1.0).sum()                                        # 统计裁剪后剩余的通道数量
    
    while remain_chs.item() < min_channel:                                           # 这里为了保证剩余超过2个通道
        bn_weights_list = list(bn_module.weight.data.abs().clone().detach().cpu().numpy())
        bn_weights_list = sorted(bn_weights_list, reverse=True)
        threshold = bn_weights_list[min_channel-1]
        mask = bn_module.weight.data.abs().ge(threshold).float()
        remain_chs = (mask == 1.0).sum()
        
    return mask


# 获得需要裁剪的BN层
def acquire_need_prune_bn(model):
    need_prune_bn = dict()                                                  # 存储需要裁剪的BN层
    ignore_bn_list = SparseBN(model).ignore_bn_list                         # 不剪枝的BN层列表
    for key, m in model.named_modules():                                    # 遍历模型所有的网络层
        if isinstance(m, nn.BatchNorm2d) and key not in ignore_bn_list:     # 如果不在忽略的BN列表
            need_prune_bn[key] = m                                          # 记录需要裁剪的BN
    return need_prune_bn, ignore_bn_list


# 获得BN层的每个通道对应的掩码
def acquire_bn_mask_dict(model, ignore_bn_list, bn_thre):
    bn_mask_dict = {}                                                                     # 存储所有的BN层的掩码
    remain_num = 0                                                                        # 记录保存的BN层数目
    for name, layer in model.named_modules():                                             # 遍历模型每一个网络层
        if isinstance(layer, nn.BatchNorm2d):                                             # 如果是BN层
            bn_mask = obtain_bn_mask(layer, bn_thre)                                      # 根据阈值生成BN掩码
            if name in ignore_bn_list:                                                    # 如果BN不裁剪
                bn_mask = torch.ones(layer.weight.data.size(), device=bn_mask.device)     # 重新生成BN掩码
            else:
                remain_num += int(bn_mask.sum())                                          # 保留的BN层的数目
            bn_mask_dict[name] = bn_mask                                                  # BN层对应的掩码
            
            # 修改BN权重
            layer.weight.data.mul_(bn_mask)                                               # 修改模型BN权重
            layer.bias.data.mul_(bn_mask)                                                 # 修改模型BN偏置
            
    return bn_mask_dict, remain_num



# 从稀疏训练模型拷贝权重到裁剪的模型
def copy_weights_to_pruned(origin_model, pruned_model, mask_bn_dict):
    origin_model_state = origin_model.state_dict()                                      # 稀疏训练模型的状态字典
    from_to_map = pruned_model.from_to_map                                              # BN层的输入层的名称
    changed_state = []
    for ((layername, layer), (pruned_layername, pruned_layer)) in zip(origin_model.named_modules(), pruned_model.named_modules()):
        assert layername == pruned_layername, layername != pruned_layername
        
        # 开始给卷积层权重赋值,不包括检测层卷积
        if isinstance(layer, nn.Conv2d) and not layername.startswith("model.24"):
            cur_bn_name = layername[:-4] + "bn"                                                             # 当前卷积的BN的名称
            if cur_bn_name in from_to_map.keys():                                                           # 当前卷积BN存在对应的前一层BN的名称
                former_bn_name = from_to_map[cur_bn_name]                                                   # 获取当前卷积输入的上一层BN的名称
                if isinstance(former_bn_name, str):                                                         # 如果是字符串，表示输入来自一个卷积层BN的输出
                    out_idx = np.squeeze(np.argwhere(mask_bn_dict[cur_bn_name].cpu().numpy()))              # 卷积的输出权重的索引
                    in_idx = np.squeeze(np.argwhere(mask_bn_dict[former_bn_name].cpu().numpy()))            # 卷积的输入权重的索引
                    
                    select_w = layer.weight.data[:, in_idx, :, :].clone()                                   # 根据输入索引选出来的权重
                    if len(select_w.shape) == 3:                                                            # 输入仅仅保留一个通道
                        select_w.unsqueeze(1)                                                               # 扩充一个维度
                    select_w = select_w[out_idx, :, :, :].clone()                                           # 根据输出索引选出来的权重
                    pruned_layer.weight.data = select_w.clone()                                             # 为剪枝过的模型赋值
                    
                    changed_state.append(layername + ".weight")
                    
                if isinstance(former_bn_name, list):                                                        # 拼接以后执行卷积操作
                    origin_in = [origin_model_state[i + ".weight"].shape[0] for i in former_bn_name]        # 原始的输入通道数
                    former_in = []                                                                          # 保留的输入索引
                    for j in range(len(former_bn_name)):
                        bn_name = former_bn_name[j]
                        tmp_idx = [i for i in range(mask_bn_dict[bn_name].shape[0]) if mask_bn_dict[bn_name][i] == 1]
                        if j > 0:
                            tmp_idx = [k + sum(origin_in[:j]) for k in tmp_idx]
                        former_in.extend(tmp_idx)
                    out_idx = np.squeeze(np.argwhere(mask_bn_dict[cur_bn_name].cpu().numpy()))
                    select_w = layer.weight.data[:, former_in, :, :].clone()
                    select_w = select_w[out_idx, :, :, :].clone()
                    
                    assert len(select_w.shape) == 4
                    pruned_layer.weight.data = select_w.clone()
                    
                    changed_state.append(layername + ".weight")                                                 # 记录赋值过的权重名称
            else:
                # 当前层BN名称不在from_to_map中,也就是第一层卷积, 卷积权重的size=[out, in, k, k]
                # 这里可以优化合并,将输入通道加入到from_to_map即可
                out_idx = np.squeeze(np.argwhere(mask_bn_dict[cur_bn_name].cpu().numpy()))                  # 输出权重对应的索引
                select_w = layer.weight.data[out_idx, :, :, :].clone()                                      # 获取要赋值的权重
                assert len(select_w.shape) == 4                                                             # 确保取出的权重形状正确
                pruned_layer.weight.data = select_w.clone()                                                 # 为剪枝过的模型赋值
                changed_state.append(layername + ".weight")                                                 # 记录赋值过的权重名称
            
        if isinstance(layer, nn.BatchNorm2d):                                                               # 所有BN层参数根据索引直接赋值
            out_idx = np.squeeze(np.argwhere(mask_bn_dict[layername].cpu().numpy()))                        # BN保留的索引
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()                                   # gamma参数
            pruned_layer.bias.data   = layer.bias.data[out_idx].clone()                                     # beta参数
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()                                 # 均值
            pruned_layer.running_var = layer.running_var[out_idx].clone()                                   # 方差
            
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")
            changed_state.append(layername + ".running_mean")
            changed_state.append(layername + ".running_var")
            changed_state.append(layername + ".num_batches_tracked")

        if isinstance(layer, nn.Conv2d) and layername.startswith("model.24"):                               # 检测头的卷积层
            former_bn_name = from_to_map[layername] 
            in_idx = np.squeeze(np.argwhere(mask_bn_dict[former_bn_name].cpu().numpy()))                    # 输入层权重索引
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :].clone()                           # 赋值卷积权重 
            pruned_layer.bias.data = layer.bias.data.clone()                                                # 赋值卷积偏置
            
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")
    
    return changed_state


# 保存剪枝后的模型到本地
def save_model_to_disk(pruned_model, file_name):
    pruned_model.eval()
    ckpt = {
        "epoch": 0, 
        "best_map": 0.0,
        "model": pruned_model,
        "optimizer": None,
        "date": datetime.now().isoformat() 
    }
    torch.save(ckpt, file_name)
    del ckpt













#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   channel_prune.py
@Time    :   2022/12/13 16:25:22
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   对yolov5执行通道剪枝
'''

'''
    通道剪枝的一般步骤
        1. 稀疏训练-train-sparse
        2. 通道裁剪-channel_prune
        3. 重新训练-finetune
'''

import os
import sys
import yaml
import torch
import numpy as np
import torch.nn as nn

sys.path.append("..")


from models.common import Conv, Bottleneck
from models.yolo import YoloV5, YoloV5Pruned, Detect
from prune.prune_tools import gather_bn_weights, obtain_bn_mask

from utils.evaluate_coco import estimate as new_estimate_coco


def run_prune(percent=0.0):
    
    cfg_file = "/workspace/yolov5-pro/models/yolov5s.yaml"
    
    model = YoloV5(cfg_file).cuda()
    ckpt = torch.load("/workspace/yolov5-pro/runs/train/exp1/weights/best.pt")
    model.load_state_dict(ckpt["model"], strict=True)
    
    model.eval()
    
    
    # --------------------------------------prune model----------------------------------------------
    # --------------------------------------prune model----------------------------------------------
    # print("model.module_list:",model.named_children())
    
    model_bn_dict = dict()                                                  # 存储需要裁剪的BN层                               
    ignore_bn_list = []                                                     # 忽略剪枝的BN层
    
    for key, m in model.named_modules():
        if isinstance(m, Bottleneck):
            if m.add:
                ignore_bn_list.append(key.rsplit(".",2)[0]+".cv1.bn")
                ignore_bn_list.append(key + '.cv1.bn')
                ignore_bn_list.append(key + '.cv2.bn')
                
        if isinstance(m, torch.nn.BatchNorm2d) and key not in ignore_bn_list:
            model_bn_dict[key] = m
            # print(key, m)
    
    print("===================================================")
    # 这些是需要裁剪的部分
    model_bn_dict = {k: v for k, v in model_bn_dict.items() if k not in ignore_bn_list}               # 这里主要是去除C3模块中的cv1的BN
    # print("Pruned BN module: ", model_bn_dict.keys())
    prune_conv_list = [layer.replace("bn", "conv") for layer in model_bn_dict.keys()]
    
    # print("Pruned conv module: ", prune_conv_list)


    print("=" * 94)
    print("=" * 94)
    
    
    # 需要裁剪的是9248层
    bn_weights, bn_bias = gather_bn_weights(model_bn_dict)                                          # 获取正则化后的BN参数
    sorted_bn_weights = torch.sort(bn_weights)[0]                                                   # 排序BN参数
    
    
    # 裁剪BN的阈值, 低于阈值的需要被裁剪掉
    thre_idx = int(len(sorted_bn_weights) * percent)
    threshold = sorted_bn_weights[thre_idx]
    
    
    # 原始模型的状态字典
    origin_model_state = model.state_dict()
    
    
    
    # --------------------------------------save pruned model config yaml----------------------------
    # --------------------------------------save pruned model config yaml----------------------------
    
    
    remain_num = 0
    mask_bn_dict = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            bn_mask = obtain_bn_mask(layer, threshold)
            if name in ignore_bn_list:
                bn_mask = torch.ones(layer.weight.data.size(), device=bn_mask.device)
            else:
                remain_num += int(bn_mask.sum())
            mask_bn_dict[name] = bn_mask
            
            # 修改BN权重
            layer.weight.data.mul_(bn_mask)
            layer.bias.data.mul_(bn_mask)
            
    real_percent = 1 - remain_num / int(len(sorted_bn_weights))    
    print("pruned percent = ", real_percent)   
    
    pruned_model = YoloV5Pruned(mask_bn_dict, "/workspace/yolov5-pro/models/yolov5s-pruned.yaml")
    
    # 兼容性更新
    for m in pruned_model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect]:
            m.inplace = True                                                                    # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()                                               # pytorch 1.6.0 compatibility 
    
    from_to_map = pruned_model.from_to_map                                                      # 存储每一层BN的上一层的BN名称
    pruned_model_state = pruned_model.state_dict()
    
    
    assert pruned_model_state.keys() == origin_model_state.keys(), "Pruned model state_dict is different from origin model."

    
    # --------------------------------------change pruned model state dict----------------------------
    # --------------------------------------change pruned model state dict----------------------------
    
    print("change state dict...")
    changed_state = []
    for ((layername, layer), (pruned_layername, pruned_layer)) in zip(model.named_modules(), pruned_model.named_modules()):
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
            
    
    # 确认是否有参数未进行赋值
    missing = [i for i in pruned_model_state.keys() if i not in changed_state]
    assert len(missing) == 0, missing
    
    print("missing = ", missing)
    
    # --------------------------------------save pruned model----------------------------
    # --------------------------------------save pruned model----------------------------
    
    # 保存剪枝并且权重赋值过的模型
    
    pruned_model.cuda()
    pruned_model.eval()
    
     
    label_map = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",  "sheep",  "sofa",  "train", "tvmonitor"]
    
    mAP, AP50, AP75, compare_img = new_estimate_coco(pruned_model, "/workspace/datasets/PASCAL_VOC2007/VOC2007_trainval/debug.txt", prefix="debug", label_map=label_map, 
                                                             image_size=640, batch_size=16, num_workers=15, 
                                                             nms_max_output_det=30000,nms_thres=0.5, conf_thres=0.001, device="cuda")
    
    print("mAP@0.50 = ", AP50)
    print("mAP@0.75 = ", AP75)
    print("mAP@0.5:0.95 = ", mAP)
    
    import cv2
    cv2.imwrite("compare_img_0001.jpg", compare_img)
    
    
    return 0








if __name__ == "__main__":
    
    run_prune()
    pass


























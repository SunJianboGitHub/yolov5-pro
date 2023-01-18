#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_prune.py
@Time    :   2023/01/15 14:11:41
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   run yolov5 channel prune
'''


'''
    # yolov5通道剪枝的步骤
        1. 稀疏训练, 运行程序 train_sparse.py
        2. 通道剪枝, 运行程序 run_prune.py
        3. 调优训练, 运行程序 finetune_pruned.py
'''


import os
import sys
import torch
import argparse
import numpy as np  
import torch.nn as nn   


sys.path.append("..")


from models.yolo import YoloV5Pruned
from prune.prune_tools import *
from utils.evaluate_coco import estimate as new_estimate_coco


def channel_prune(opts):
    if opts.weights.endswith(".pt"):                                    # 要裁剪的模型权重
        checkpoint = torch.load(opts.weights)                           # 加载模型权重
        origin_model = checkpoint['model']                              # 获得稀疏训练好的模型
        origin_model.eval()                                             # 转变为评估模式
    else:
        return -1
    
    need_prune_bn, ignore_bn_list = acquire_need_prune_bn(origin_model) # 需要裁剪的BN层
    bn_weights, bn_biases = gather_bn_weights(need_prune_bn)            # 要裁剪的BN权重,共9248层
    sorted_bn_weights = torch.sort(bn_weights)[0]                       # 将需要裁剪的BN权重进行排序
    
    # 根据裁剪比例,获取裁剪BN的阈值
    thre_idx = int(len(sorted_bn_weights) * opts.percent)               # 根据剪枝比例,确定BN阈值索引
    bn_thre = sorted_bn_weights[thre_idx]                               # 找出BN阈值,低于阈值的进行剪枝

    bn_mask_dict, remain_num = acquire_bn_mask_dict(origin_model, ignore_bn_list, bn_thre)          # 获得所有BN层的掩码,用于构建剪枝模型
    real_percent = 1 - remain_num / int(len(sorted_bn_weights))                                     # 计算实际的剪枝比例

    print("Percent of channel prune, origin_percent = %.3f, real_percent = %.3f" %(opts.percent, real_percent))
    
    
    # 构建剪枝后的模型
    pruned_model = YoloV5Pruned(bn_mask_dict, yaml_cfg_file=opts.prune_cfg)             # 构建剪枝后的模型
    
    changed_state = copy_weights_to_pruned(origin_model, pruned_model, bn_mask_dict)    # 剪枝后模型权重赋值
    
    assert pruned_model.state_dict().keys() == origin_model.state_dict().keys(), "Pruned model state_dict is different from origin model."

    # 确认是否有参数未进行赋值
    missing = [i for i in pruned_model.state_dict().keys() if i not in changed_state]
    assert len(missing) == 0, missing
    
    pruned_model.cuda()
    pruned_model.eval()
    mAP, AP50, AP75, compare_img = new_estimate_coco(pruned_model, opts.data, prefix=opts.prefix, label_map=opts.label_map, 
                                                             image_size=opts.img_size, batch_size=opts.batch_size, num_workers=15, 
                                                             nms_max_output_det=30000,nms_thres=0.5, conf_thres=0.001, device=opts.device)
    
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    file_name = str(opts.prefix) + "_pruned_init_model.pt"
    file_name = os.path.join(opts.save_dir, file_name)
    ckpt = {
        "epoch": 0, 
        "best_map": mAP,
        "model": pruned_model,
        "optimizer": None,
        "date": datetime.now().isoformat() 
    }
    torch.save(ckpt, file_name)
    del ckpt
    
    print("Pruned init model save successfully! Please finetune pruned model...")
    
    return 0






if __name__ == "__main__":
    
    label_map = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",  "sheep",  "sofa",  "train", "tvmonitor"]
    
    
    parser = argparse.ArgumentParser()
    
    # 训练权重加载与结果保存参数,必须修改
    parser.add_argument("--prune_cfg", type=str, default="/workspace/yolov5-pro/models/yolov5s-pruned.yaml", help="Network config file.")           # 剪枝模型的配置文件
    parser.add_argument("--data", type=str, default="/workspace/datasets/PASCAL_VOC2007/VOC2007_test/test_yolo.txt", help="Test data for pruned model.")                                                         # 为了验证剪枝后模型效果的测试集
    parser.add_argument("--weights", type=str, default="/workspace/yolov5-pro/runs/train/exp4/weights/best.pt", help="load pretrained weights.")    # 稀疏训练模型的权重
    parser.add_argument("--save_dir", type=str, default="/workspace/yolov5-pro/runs/train/exp4/pruned", help="Save pruned model dir.")
    parser.add_argument("--percent", type=float, default=0.50, help="Percent of channel prune.")                                                    # 通道剪枝的比例
    
    parser.add_argument("--img_size", type=int, default=640, help="Percent of channel prune.")    # 通道剪枝的比例
    parser.add_argument("--batch_size", type=int, default=16, help="Percent of channel prune.")    # 通道剪枝的比例
    parser.add_argument("--prefix", type=str, default="VOC2007", help="Percent of channel prune.")    # 通道剪枝的比例
    
    parser.add_argument("--device", type=int, default=0, help="Percent of channel prune.")    # 通道剪枝的比例
    parser.add_argument("--num_workers", type=int, default=15, help="Percent of channel prune.")    # 通道剪枝的比例
    
    opts = parser.parse_args()
    
    opts.label_map = label_map
    
    
    channel_prune(opts)




























#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_sparse.py
@Time    :   2023/01/12 16:50:03
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   yolov5模型的稀疏训练, L1正则化BN权重
'''


'''
    # 稀疏BN
    # 对需要剪枝的卷积的BN层执行L1正则化

'''

import os
import sys
import yaml
import torch
import argparse
import torchvision

import numpy as np

from tqdm import tqdm
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]                                  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                          # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))          # relative


from utils.train_tools import *
from utils.general import setup_random_seed
from utils.general import _single_instance_logger as logger
from prune.sparse_tools import SparseBN



# yolov5 稀疏训练
def train_sparse(opts, hyps):
    
    # 随机布种
    setup_random_seed(3)
    
    # 数据集采用标准化操作,均值和方差采用COCO
    MEAN, STD = acquire_mean_std(data_type="coco")
    
    # 初始化日志记录器、保存权重文件名
    best_pt, last_pt, tbd_logs = basic_initialization(opts.save_dir)
    
    # 规范化超参数(按照batch_size=64进行规范化操作)
    nbs = 64                                                                # 模型的训练过程都需要根据bs=64规范化
    accumulate = max(round(nbs / opts.batch_size), 1)                       # 累计几次达到nbs
    hyps['weight_decay'] *= opts.batch_size * accumulate / nbs              # 规范化超参数
    
    # 初始化模型、优化器、调度器
    model, optimizer, scheduler, lf = model_initialization(opts, hyps)
    
    
    # 断言
    assert len(opts.label_map) == model.num_classes
    
    # 加载预训练权重或者恢复训练
    start_epoch, best_map = pretrained_and_resume(model, optimizer, opts)
    
    # 初始化滑动平均模型,切记放在加载预训练权重之后或者恢复训练之后
    ema_model = ema_model_initialization(model)
    
    # 规范化loss权重系数的超参数
    normalize_loss_hyps(model, opts, hyps)
    
    # 损失函数
    loss_func = acquire_loss_func(model, opts, hyps, box_reg="SIoU")
    
    # 获取数据集路径
    train_path, val_path, test_path = acquire_datasets_path(data_yaml=opts.data)
    
    
    # 稀疏化BN
    if opts.sr > 0:
        optimizer_bn = SparseBN(model, sr=opts.sr, epochs=opts.epochs)
    else:
        optimizer_bn = None
    
    # 训练前准备
    train_dataloader = acquire_dataloader(opts, hyps, data_path=train_path, shuffle=True)
    num_batch = len(train_dataloader)
    warmup_iters = max(round(hyps['warmup_epochs'] * num_batch), 500)
    scheduler.last_epoch = start_epoch - 1
    
    # --------------------------------Start Training----------------------------------
    # --------------------------------Start Training----------------------------------
    mAP, ap50, ap75 = 0, 0, 0
    for epoch in range(start_epoch, opts.epochs):
        logger.info(('\n' + '%11s' * 9) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'num_objs', 'matched', 'unmatched', 'Size'))
        pbar = tqdm(enumerate(train_dataloader), total=num_batch, ncols=NCOLS, bar_format=BAR_FORMAT)
        mloss = torch.zeros(3, device=opts.device)
        
        model.train()
        optimizer.zero_grad()
        for i, (imgs, targets, visual_info) in pbar:
            num_iters =  i + num_batch * epoch
            if num_iters < warmup_iters:                                # 学习率预热
                accumulate = max(1, np.interp(num_iters, [0, warmup_iters], [1, nbs / opts.batch_size]).round())
                learning_rate_warmup(optimizer, hyps, lf, epoch, num_iters, warmup_iters)
    
            # 前向推理,计算loss
            imgs, targets = imgs.to(opts.device).float() / 255, targets.to(opts.device)
            if MEAN and STD:
                imgs = torchvision.transforms.Normalize(mean=MEAN, std=STD)(imgs)                       # 输入处理为标准正态分布,原始的yolov5没有该操作
            predicts = model(imgs)
            total_loss, loss_items, (matched_num_mean, unmatched_num_target) = loss_func(predicts, targets)
            
            # 反向传播, 计算梯度
            total_loss.backward()
            
            
            # 梯度更新
            # ---------------------------Sparse BN---------------------------
            # ---------------------------Sparse BN---------------------------
            # L1正则化优化BN, 直接手动修改梯度
            if opts.sr > 0:
                optimizer_bn.add_bn_grad(epoch=epoch)
            # ---------------------------Sparse BN---------------------------
            # ---------------------------Sparse BN---------------------------
            
            
            # 累加之后,进行梯度更新
            if num_iters % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ema_model is not None:
                    ema_model.update(model)

            mloss = (mloss * i + loss_items) / (i + 1)                  # 更新平均损失
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%11s' * 2 + '%11.4g' * 7) % (f'{epoch}/{opts.epochs - 1}', mem, *mloss, targets.shape[0], 
                                                                matched_num_mean, unmatched_num_target, imgs.shape[-1]))
    
            # tensorboard记录训练loss
            record_train_loss(tbd_logs, mloss, num_iters)
            
            # tensorboard检验训练输入图像
            if i ==0:
                record_train_image(tbd_logs, visual_info, epoch)

        # ---------------------------Show BN---------------------------
        # ---------------------------Show BN---------------------------
        # 显示BN层的权重与偏置的分布
        if opts.sr > 0:
            bn_weights, bn_biases = optimizer_bn.bn_distribution()
            record_bn_distribution(tbd_logs, bn_weights, bn_biases, epoch=epoch)
        # ---------------------------Show BN---------------------------
        # ---------------------------Show BN---------------------------
        
        
        # 到此,训练完一个完整的epoch
        # 记录当前epoch的学习率,并更新学习率
        record_learning_rate(tbd_logs, optimizer, epoch)
        scheduler.step()
        
        # --------------------------------Evaluation Model----------------------------------
        # --------------------------------Evaluation Model----------------------------------
        if epoch % 1 == 0:
            mAP, ap50, ap75, compare_img = acquire_evaluate_result(model=ema_model.ema, opts=opts, data_path=val_path, normalize=(MEAN, STD))
            
            # 控制台tqdm显示训练过程
            logger.info(('%13s' * 4) % ('Epoch', 'AP@0.5', 'AP@0.75', 'mAP'))
            logger.info(('%13s' * 1 + '%13.4g' * 3) % (f'{epoch}/{opts.epochs - 1}', ap50, ap75, mAP))
            
            # 记录评估阶段的推理图像
            record_infer_image(tbd_logs, compare_img, epoch)
            
            # 记录评估阶段的mAP
            record_evaluate_mAP(tbd_logs, ap50=ap50, ap75=ap75, mAP=mAP, epoch=epoch)
            
            # --------------------------------Save best Model----------------------------------
            # --------------------------------Save best Model----------------------------------
            # 保存最好的模型到本地硬盘
            if best_map < mAP:
                best_map = mAP
                save_model_to_disk(best_pt, ema_model.ema, optimizer, epoch, best_map=best_map)
    
    # --------------------------------Save last Model----------------------------------
    # --------------------------------Save last Model----------------------------------
    # 评估保存最后一轮的模型
    save_model_to_disk(last_pt, ema_model.ema, optimizer, epoch, best_map=mAP)
    
    # 评估最好模型在测试集上的mAP
    load_weights_to_model(model, best_pt, opts.device)
    mAP, ap50, ap75, compare_img = acquire_evaluate_result(model=model, opts=opts, data_path=test_path, normalize=(MEAN, STD))
    logger.info(('%13s' * 4) % ('dataset', 'AP@0.5', 'AP@0.75', 'mAP'))
    logger.info(('%13s' * 1 + '%13.4g' * 3) % ("test_data", ap50, ap75, mAP))
    pass


    return 0



if __name__ == "__main__":
    
    # 这里根据数据集需要修改
    label_map = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",  "sheep",  "sofa",  "train", "tvmonitor"]                     # 检测类别标签
    
    excavator_map = ["excavator"]
    
    parser = argparse.ArgumentParser()
    
    # 训练配置文件，根据数据集必须修改的
    parser.add_argument("--cfg", type=str, default="/workspace/yolov5-pro/models/yolov5l.yaml", help="mode.yaml path.")                         # 模型配置文件, 主要修改检测类别数、anchors
    parser.add_argument('--data', type=str, default='/workspace/yolov5-pro/data/config/VOC2007.yaml', help='data.yaml path')                    # 数据集配置文件, 主要修改各个数据集路径
    parser.add_argument('--hyps', type=str, default='/workspace/yolov5-pro/data/hyps/hyp.VOC2007-base.yaml', help='hyp.yaml path')             # 超参数配置文件, 主要修改学习率、损失函数权重
    parser.add_argument("--prefix", type=str, default="VOC2007", help="Prefix is used mark datasets.")                                          # 前缀字符串, 主要修改为数据集名称
    
    # 训练权重加载与结果保存参数,必须修改
    parser.add_argument("--weights", type=str, default="", help="load pretrained weights.")                      # 预训练权重的路径
    parser.add_argument('--resume', type=str, default="", help='resume checkpoints.')                                                           # 检查点文件的路径
    parser.add_argument("--save_dir", type=str, default="/workspace/yolov5-pro/runs/train/exp4", help="Save weights and logs.")                 # 日志文件的保存路径
    
    # 稀疏训练系数
    parser.add_argument("--sr", type=float, default=0.001, help="L1 normalization scale.")
    
    
    
    # 训练基础参数,可适当修改
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--cosine_lr", type=bool, default=True)
    
    # 训练图像增强参数,可适当修改
    parser.add_argument("--augment", type=bool, default=True, help="Only mosaic")
    parser.add_argument("--mixed_aug", type=bool, default=True, help="Mosaic and center scale")
    parser.add_argument("--mosaic_num", type=list, default=[4, 9], help="You can set [4, 9, 16, 25]")
    parser.add_argument("--cache_images", type=bool, default=False, help="Load all images to RAM.")
    
    
    # 训练设备相关参数,几乎不需要修改
    parser.add_argument("--num_workers", type=int, default=16, help="Number of threads")
    parser.add_argument("--device", type=int, default=0, help="GPU ID")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank.")
    
    
    
    opts = parser.parse_args()
    opts.label_map = label_map
    # opts.label_map = excavator_map
    
    with open(opts.hyps, "r") as f:
        hyps = yaml.load(f, Loader=yaml.FullLoader)
    
    train_sparse(opts, hyps)
    
    pass






























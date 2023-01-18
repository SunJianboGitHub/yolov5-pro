#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_tools.py
@Time    :   2022/12/30 16:11:32
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   some train tools
'''

import os
import yaml
import torch
import numpy as np
from datetime import datetime

from models.yolo import YoloV5
from utils.loss import ComputeLoss
from utils.plots import draw_norm_bboxes
from utils.dataset import create_dataloader
from utils.tensorbord_utils import LogRecoder
from utils.evaluate import estimate as estimate_self
from utils.evaluate_coco import estimate as estimate_coco
from utils.torch_utils import smart_optimizer, smart_resume, acquire_lr_scheduler, load_pretrained_weights, ModelEMA


# tqdm格式化显示
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
NCOLS = 0 if True else shutil.get_terminal_size().columns               # 在docker中,将ncols设置为0,就不会出现格式混乱




# 根据数据类型获取均值和方差
def acquire_mean_std(data_type="coco"):
    mean, std = None, None
    if data_type.lower() == "coco":
        mean = [0.471, 0.448, 0.408]       # COCO上的图像均值, RGB通道
        std  = [0.234, 0.239, 0.242]       # COCO上的图像标准差, RGB通道
    elif data_type.lower() == "voc2007":
        pass
    elif data_type.lower() == "voc2012":
        pass
    else:
        mean = [0.485, 0.456, 0.406]   # ImageNet上的图像均值, RGB通道
        std  = [0.229, 0.224, 0.225]   # ImageNet上的图像标准差, RGB通道
        
    return mean, std


# 获取数据集迭代器
def acquire_dataloader(opts, hyps, data_path, shuffle=True):
    dataloader, _ = create_dataloader(datas_path=data_path,                                         # 数据集的路径
                                         hyp=hyps,                                                  # 超参数，暂时没用到
                                         shuffle=shuffle,                                           # 打乱数据集，训练集打乱，测试集不打乱                                 
                                         augment=opts.augment,                                      # 基础的图像增强
                                         mixed_aug=opts.mixed_aug,                                  # 马赛克增强和中心缩放
                                         cache_images=opts.cache_images,                            # 是否缓存图像，根据数据集大小确定是否缓存
                                         mosaic_nums=opts.mosaic_num,                               # 马赛克增强的方法，也就是多少个图片进行镶嵌增强
                                         prefix=opts.prefix,                                        # 前缀，用于区分数据集
                                         batch_size=opts.batch_size,                                # 网络的输入批次大小
                                         img_size=opts.img_size,                                    # 网络的输入尺寸
                                         num_workers=opts.num_workers,                              # 读取数据集，开启的线程数
                                         border_fill_value=114)                                     # 图像边界的填充数值
    
    
    return dataloader


# 获得损失函数
def acquire_loss_func(model, opts, hyps, box_reg="SIoU"):
    loss_func = ComputeLoss(num_classes=model.num_classes, anchors=model.anchors, 
                            hyp=hyps, device=opts.device, box_reg=box_reg)
    return loss_func


# 获取数据集的路径
def acquire_datasets_path(data_yaml):
    with open(data_yaml, "r") as f:
        data_path = yaml.load(f, Loader=yaml.FullLoader)
    return data_path['train_path'], data_path['val_path'], data_path['test_path']



# 是否混合精度训练
def is_mixed_precision():
    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
    except:
        print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
        mixed_precision = False  # not installed

    return mixed_precision


# 初始化日志记录器、保存文件
def basic_initialization(save_dir):
    log_dir = os.path.join(save_dir, "logs")                       # 日志的保存目录
    weight_dir = os.path.join(save_dir, "weights")                 # 权重的保存目录
    os.makedirs(log_dir, exist_ok=True)                            # 如果目录不存在,创建多级目录
    os.makedirs(weight_dir, exist_ok=True)                         # 如果目录不存在,创建多级目录
    
    last_pt = weight_dir + os.sep + "last.pt"                      # 保存最后的权重文件名称
    best_pt = weight_dir + os.sep + "best.pt"                      # 保存的最好的模型权重文件名称
    
    tbd_logs = LogRecoder(log_dir=log_dir, flush_secs=10)          # tensorboard日志记录器
    
    return best_pt, last_pt, tbd_logs


# 模型、优化器、学习率调度器初始化
def model_initialization(opts, hyps):
    model = YoloV5(yaml_cfg_file=opts.cfg).to(opts.device)                                                                        # 构建模型
    optimizer = smart_optimizer(model, name=opts.optimizer, lr=hyps["lr0"], momentum=hyps["momentum"], decay=hyps['weight_decay'])      # 智能优化器,参数分组优化
    scheduler, lf = acquire_lr_scheduler(optimizer, lrf=hyps['lrf'], epochs=opts.epochs, cos_lr=opts.cosine_lr, T=100, decay_rate=0.9)         # 余弦退火+周期性重启
    return model, optimizer, scheduler, lf



# 加载预训练权重或者恢复训练
def pretrained_and_resume(model, optimizer, opts):
    
    # 加载预训练权重到模型,这里只是加载参数
    if opts.weights != "":
        load_pretrained_weights(model, opts.weights, opts.device)
    
    # 根据检查点文件,恢复模型训练,不仅恢复模型参数,还包括优化器、学习率、epoch等等
    start_epoch, best_map = 0, 0.0
    if opts.resume != "":
        start_epoch, best_map = smart_resume(model, optimizer, opts.resume, opts.device, epochs=opts.epochs)
    
    return start_epoch, best_map


# 初始化滑动平均模型,切记放在加载预训练权重之后或者恢复训练之后
def ema_model_initialization(model):
    ema_model = ModelEMA(model)
    return ema_model


# 规范化loss权重超参数
def normalize_loss_hyps(model, opts, hyps):
    nl = len(model.anchors)                                                # 检测头的数量，通常是三个检测头
    nc = model.num_classes                                                 # 检测类别数
    hyps["box"] *= 3 / nl                                                  # 对层进行缩放，因为设置的是针对三层的损失系数
    hyps["cls"] *= (nc / 80 ) * (3 / nl)                                   # 对类别和层进行适当缩放
    hyps["obj"] *= (opts.img_size / 640) ** 2 * (3 / nl)                   # 缩放到 image size and layers,这里的平方是因为obj是针对像素的
    

# 学习率预热
def learning_rate_warmup(optimizer, hyps, lf, epoch, num_iters, warmup_iters):
    if num_iters < warmup_iters:
        xi = [0, warmup_iters]
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = np.interp(num_iters, xi, [hyps['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(num_iters, xi, [hyps['warmup_momentum'], hyps['momentum']])
    pass


# tensorboard记录训练loss
def record_train_loss(tbd_logs, mloss, num_iters):
    mloss = mloss.cpu().numpy()
    # 记录总loss,包括类别损失、置信度损失、边界框损失
    tbd_logs.add_line(tag_name="loss/total_loss", 
                                   scaler_value=np.sum(mloss), 
                                   global_step=num_iters)
    # 记录单独的loss
    tbd_logs.add_lines(tag_name="loss/single_loss", 
                                    tag_scalar_dict={
                                        "box_loss": mloss[0],
                                        "cls_loss": mloss[1],
                                        "obj_loss": mloss[2]}, 
                                    global_step=num_iters)
    pass


# tensorboard记录显示训练输入图像
def record_train_image(tbd_logs, visual_info, epoch=0):
    _, show_img, norm_label = visual_info
    draw_norm_bboxes(show_img, norm_label[:, 1:], color=(0, 255, 0), thickness=1)
    show_img = np.transpose(show_img[..., ::-1], (2, 0, 1))
    tbd_logs.add_image(tag_name="image/verify_input", img=show_img, global_step=epoch, walltime=None, dataformats="CHW")


# 记录当前epoch的学习率
def record_learning_rate(tbd_logs, optimizer, epoch):
    lr = [x['lr'] for x in optimizer.param_groups]
    tbd_logs.add_line(tag_name="learning_rate", scaler_value=lr[1], global_step=epoch)


# tensorboard记录评估阶段的推理结果
def record_infer_image(tbd_logs, compare_img, epoch=0):
    tbd_logs.add_image(tag_name="image/infer_result", img=compare_img, global_step=epoch, walltime=None, dataformats="CHW")
    

# tensorboard记录评估的mAP
def record_evaluate_mAP(tbd_logs, ap50, ap75, mAP, epoch):
    tbd_logs.add_lines(tag_name="evaluate/mAP", 
                                    tag_scalar_dict={
                                        "mAP@0.50": ap50,
                                        "mAP@0.75": ap75,
                                        "mAP@0.5:0.95": mAP}, global_step=epoch)


def record_bn_distribution(tbd_logs, bn_weights, bn_biases, epoch):
    tbd_logs.add_histogram(tag_name="distribution/st_bn_weights", values=bn_weights, global_step=epoch, bins="doane")
    tbd_logs.add_histogram(tag_name="distribution/st_bn_bias", values=bn_biases, global_step=epoch, bins="doane")
    pass



# COCO API获得评估结果
def acquire_evaluate_result(model, opts, data_path, normalize=None):
    # 自己写的mAP计算过程
    # AP50, AP75, mAP, compare_img = estimate_self(model, opts.val_path, method="interp11", num_cls=model.num_classes, image_size=opts.img_size, batch_size=opts.batch_size, 
    #                                 num_workers=opts.num_workers)
    
    # 基于COCO API的mAP计算过程
    mAP, AP50, AP75, compare_img = estimate_coco(model, data_path, prefix=opts.prefix, label_map=opts.label_map, 
                                                             image_size=opts.img_size, batch_size=opts.batch_size, num_workers=opts.num_workers, 
                                                             nms_max_output_det=30000,nms_thres=0.5, conf_thres=0.001, device=opts.device, normalize=normalize)
    # 输出的图像是OpenCV格式的,也就是HWC BGR的
    # 我们需要转换为 CHW RGB的
    compare_img = np.transpose(compare_img[..., ::-1], (2, 0, 1))
    
    return mAP, AP50, AP75, compare_img


# 加载模型权重参数到模型
def load_weights_to_model(model, best_pt, device):
    if best_pt.endswith(".pt"):
        load_pretrained_weights(model, best_pt, device)
    



# torch.save保存模型到本地
def save_model_to_disk(file_name, model, optimizer, epoch, best_map):
    # 如果是最后一轮就保存最后一轮的mAP
    ckpt = {
        "epoch": epoch, 
        "best_map": best_map,
        "model": model,
        "optimizer": optimizer.state_dict(),
        "date": datetime.now().isoformat() 
    }
    torch.save(ckpt, file_name)
    del ckpt
    

# 保存裁剪过(通道剪枝)的模型到本地
def save_pruned_model_to_disk(file_name, model, optimizer, epoch, best_map):
    # 如果是最后一轮就保存最后一轮的mAP
    ckpt = {
        "epoch": epoch, 
        "best_map": best_map,
        "model": model,
        "optimizer": optimizer.state_dict(),
        "date": datetime.now().isoformat() 
    }
    torch.save(ckpt, file_name)
    del ckpt














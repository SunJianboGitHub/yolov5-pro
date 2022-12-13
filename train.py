#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_yolov5.py
@Time    :   2022/10/12 09:37:47
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   train model in yolov5
'''

import os
import sys
import time
import torch
import visdom
import argparse
import numpy as np
from pathlib import Path

from tqdm import tqdm


import torch.optim as optim

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# sys.path.append(os.getcwd())



from models.yolo import YoloV5
from utils.dataset import create_dataloader
from utils.loss import ComputeLoss
from utils.plots import draw_norm_bboxes
from utils.evaluate import estimate as estimate_self
from utils.coco_evauate import estimate as estimate_coco
from utils.new_evaluate import estimate as new_estimate_coco
from utils.general import setup_random_seed
from utils.general import _single_instance_logger as logger
from utils.torch_utils import smart_optimizer, smart_resume, acquire_lr_scheduler, load_pretrained_weights, ModelEMA

from utils.tensorbord_utils import LogRecoder                       # 用于TensorBoard记录显示日志
from utils.sparse_bn import OptimizerBN                             # 用于稀疏化训练,也就是通道剪枝训练                      




BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format



mixed_precision = False
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed



def train(args, hyp):
    
    log_dir = os.path.join(args.save_dir, "logs")
    weight_dir = os.path.join(args.save_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    
    last_pt = weight_dir + os.sep + "last.pt"
    best_pt = weight_dir + os.sep + "best.pt"
    
    results_file = log_dir + os.sep + "results.txt"
    
    
    # 随机布种
    setup_random_seed(3)
    
    # 创建模型、优化器、学习率策略
    model = YoloV5(yaml_cfg_file=args.model_cfg).to(args.device)                                                                                        # 搭建模型
    
    
    
    # 优化器
    nbs = 64                                                                                                                                            # nominal batch size
    accumulate = max(round(nbs / args.batch_size), 1)                                                                                                   # accumulate loss before optimizing
    hyp['weight_decay'] *= args.batch_size * accumulate / nbs                                                                                           # scale weight_decay
    optimizer = smart_optimizer(model, name=args.optimizer, lr=hyp["lr0"], momentum=hyp["momentum"], decay=hyp["weight_decay"])                         # 智能优化器
    scheduler, lf = acquire_lr_scheduler(optimizer, hyp["lrf"], epochs=args.epochs, T=args.epochs, cos_lr=args.cos_lr)                                  # 学习率scheduler,最终学习率通过超参数lrf设置
    
    ema_model = ModelEMA(model)                                                                                                                         # 模型的指数滑动平均
    
    # 加载预训练权重到模型, 这里只是加载参数
    if args.weights != "":
        load_pretrained_weights(model, args.weights, args.device)                                                                                       # 加载预训练权重到模型,这里不会加载anchors参数
    
    # 根据检查点文件,恢复模型训练,不仅恢复模型参数,还有优化器、学习率、epoch等等。
    start_epoch, best_map = 0, 0.0                                                                                                                      # 起始epoch，最好的mAP
    if args.resume != "":
        start_epoch, best_map = smart_resume(model, optimizer, args.resume, results_file, args.device, ema=None, epochs=args.epochs)
    
    
    # Mixed precision training https://github.com/NVIDIA/apex
    is_amp = False
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        is_amp = True
    
    
    train_dataloader, _ = create_dataloader(datas_path=args.train_path,                                # 数据集的路径
                                         hyp=None,                                                  # 超参数，暂时没用到
                                         shuffle=True,                                              # 打乱数据集，训练集打乱，测试集不打乱                                 
                                         augment=args.augment,                                      # 基础的图像增强
                                         mixed_aug=args.mixed_aug,                                  # 马赛克增强和中心缩放
                                         cache_images=args.cache_images,                            # 是否缓存图像，根据数据集大小确定是否缓存
                                         mosaic_nums=args.mosaic_num,                               # 马赛克增强的方法，也就是多少个图片进行镶嵌增强
                                         prefix=args.prefix,                                        # 前缀，用于区分数据集
                                         batch_size=args.batch_size,                                # 网络的输入批次大小
                                         img_size=args.img_size,                                    # 网络的输入尺寸
                                         num_workers=args.num_workers,                              # 读取数据集，开启的线程数
                                         border_fill_value=114)                                     # 图像边界的填充数值
    
    # 模型参数
    nl = len(model.anchors)                                                                         # 检测头的数量，通常是三个检测头
    nc = model.num_classes
    hyp["box"] *= 3 / nl                                                                            # 对层进行缩放，因为设置的是针对三层的损失系数
    hyp["cls"] *= (nc / 80 ) * (3 / nl)                                                             # 对类别和层进行适当缩放
    hyp["obj"] *= (args.img_size / 640) ** 2 * (3 / nl)                                             # 缩放到 image size and layers,这里的平方是因为obj是针对像素的
    
    
    # 开始启动训练
    start_time = time.time()                                                                        # 记录起始时间
    num_batch = len(train_dataloader)                                                               # 数据集一共包括多少个batch
    iters_warmup = max(round(hyp["warmup_epochs"] * num_batch), 100)                                # 学习率预热，最高100次迭代或者3个epochs
    
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=is_amp)
    
    results = (0, 0, 0, 0, 0, 0, 0)                                                                 # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    loss_func = ComputeLoss(num_classes=nc, anchors=model.anchors, hyp=hyp)
    
    
    # --------------------------------Start Training----------------------------------
    optimizer_bn = OptimizerBN(model=model, sr=hyp['sr'], epochs=args.epochs)                                                             # 用于稀疏化训练
    
    for epoch in range(start_epoch, args.epochs):                                                                                       # 开始每一轮的训练
        model.train()                                                                                                                   # 开启训练模式
        
        mloss = torch.zeros(3, device=args.device)                                                                                      # 记录平均损失
        logger.info(('\n' + '%11s' * 9) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'num_objs', 'matched_num_mean', 'unmatched_num_target', 'Size'))
        pbar = tqdm(enumerate(train_dataloader), total=num_batch, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')                        # 进度条
        
        optimizer.zero_grad()                                                                                                           # 梯度置为零
        for i, (imgs, targets, visual_info) in pbar:                                                                                     # 遍历每一批次的数据
            num_iters = i + num_batch * epoch                                                                                           # 当前的迭代次数，用于控制warmup
            
            # Warmup操作
            if num_iters < iters_warmup:                                                                                                # 如果满足warmup
                xi = [0, iters_warmup]                                                                                                  # x 插值
                accumulate = max(1, np.interp(num_iters, xi, [1, nbs / args.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(num_iters, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iters, xi, [hyp['warmup_momentum'], hyp['momentum']])
            
            imgs, targets = imgs.to(args.device).float() / 255, targets.to(args.device)
            predicts = model(imgs)
            total_loss, loss_items, (matched_num_mean, unmatched_num_target) = loss_func(predicts, targets)
            
            optimizer.zero_grad()
            total_loss.backward()
            
            
            # BN权重的L1正则化
            optimizer_bn.update_bn(epoch)
        
            optimizer.step()
            
           
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%11s' * 2 + '%11.4g' * 7) %
                                     (f'{epoch}/{args.epochs - 1}', mem, *mloss, targets.shape[0], matched_num_mean, unmatched_num_target, imgs.shape[-1]))

            show = mloss.cpu().numpy()
            
            # visdom显示loss曲线
            # args.vis.line([show[0]], [num_iters], win="train_loss", update="append", name="train_loss")
            
            # tensorboard显示loss曲线
            args.tbd_logs.add_line(tag_name="loss/total_loss", 
                                   scaler_value=total_loss.detach().cpu().item(), 
                                   global_step=num_iters)
            
            args.tbd_logs.add_lines(tag_name="loss/single_loss", 
                                    tag_scalar_dict={
                                        "box_loss": show[0],
                                        "cls_loss": show[1],
                                        "obj_loss": show[2]}, 
                                    global_step=num_iters)
            
            
            # TensorBoard显示训练的图像
            _, show_img, norm_label = visual_info
            draw_norm_bboxes(show_img, norm_label[:, 1:], color=(0, 255, 0), thickness=1)
            show_img = np.transpose(show_img[..., ::-1], (2, 0, 1))
            args.tbd_logs.add_image(tag_name="image/verify_input", img=show_img, global_step=0, dataformats="CHW")
            # args.vis.image(show_img[..., ::-1], win="visual_image")
            
            
            

        # 展示BN层权重分布
        # TensorBoard显示BN层权重分布
        bn_weights, bn_bias = optimizer_bn.display_bn()
        args.tbd_logs.add_histogram(tag_name="distribution/st_bn_weights", values=bn_weights, global_step=epoch, bins="doane")
        args.tbd_logs.add_histogram(tag_name="distribution/st_bn_bias", values=bn_bias, global_step=epoch, bins="doane")

        # args.vis.histogram(bn_weights, win="bn_weights", opts=dict(numbins = min(300, len(bn_weights))))
        
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        
        # TensorBoard显示学习率
        args.tbd_logs.add_line(tag_name="learning_rate", scaler_value=lr[0], global_step=epoch)
        
        
        if epoch > 1 and epoch % 5 ==0:
            # 验证集评估
            # AP50, AP75, mAP, compare_img = estimate_self(model, args.val_path, method="interp11", num_cls=nc, image_size=args.img_size, batch_size=args.batch_size, 
            #                         num_workers=args.num_workers)
            
            # mAP, AP50, AP75 = estimate_coco(model, args.val_path, image_size=args.img_size, 
            #                              batch_size=args.batch_size, num_workers=args.num_workers, nms_max_output_det=30000, 
            #                              nms_thres=0.5, conf_thres=0.001, device=args.device)
            
            mAP, AP50, AP75, compare_img = new_estimate_coco(model, args.val_path, prefix=args.prefix, label_map=args.label_map, 
                                                             image_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers, 
                                                             nms_max_output_det=30000,nms_thres=0.5, conf_thres=0.001, device=args.device)
            
            # 传递过来的OpenCV的格式,也就是HWC
            compare_img = np.transpose(compare_img[..., ::-1], (2, 0, 1))
            # TensorBoard显示图像
            args.tbd_logs.add_image(tag_name="image/infer_result", img=compare_img, global_step=1, dataformats="CHW")
            # args.vis.image(compare_img, win="compare_img")
            
            
            
            
            logger.info(('%11s' * 4) % ('Epoch', 'AP@0.5', 'AP@0.75', 'mAP'))
            logger.info(('%11s' * 1 + '%11.4g' * 3) % (f'{epoch}/{args.epochs - 1}', AP50, AP75, mAP))
            
            # TensorBoard显示mAP曲线,显示多个曲线
            args.tbd_logs.add_lines(tag_name="evaluate/mAP", 
                                    tag_scalar_dict={
                                        "mAP@0.50": AP50,
                                        "mAP@0.75": AP75,
                                        "mAP@0.5:0.95": mAP}, global_step=epoch)
            
            # args.vis.line([mAP], [epoch], win="mAP-new-coco", update="append", name="mAP")
            # args.vis.line([AP50], [epoch], win="mAP-new-coco", update="append", name="AP50")
            # args.vis.line([AP75], [epoch], win="mAP-new-coco", update="append", name="AP75")


    return 0







if __name__ == "__main__":

    import yaml

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_cfg", type=str, default="/workspace/yolov5-pro/models/yolov5s-v2.yaml", help="Network yaml config file.")
    parser.add_argument("--train_path", type=str, default="/workspace/datasets/PASCAL_VOC2007/VOC2007_trainval/debug.txt", help="Datasets trainval.txt path.")
    parser.add_argument("--val_path", type=str, default="/workspace/datasets/PASCAL_VOC2007/VOC2007_trainval/debug.txt", help="Datasets val.txt path.")
    
    parser.add_argument("--save_dir", type=str, default="/workspace/yolov5-pro/runs/train/exp1", help="Save weights and logs.")
    parser.add_argument('--resume', type=str, default="", help='resume from path/to/last.pt, or restart run if blank.')
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument("--epochs", type=int, default=100, help="Train epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer method.")
    parser.add_argument("--cos_lr", type=bool, default=True, help="cosine lr.")
    
    parser.add_argument("--augment", type=bool, default=False, help="Images augmentation for trainning.")
    parser.add_argument("--mixed_aug", type=bool, default=False, help="Mosaic and center scale")
    parser.add_argument("--mosaic_num", type=list, default=[4], help="You can set [4, 9, 16, 25]")
    parser.add_argument("--cache_images", type=bool, default=False, help="Load all images to RAM.")
    parser.add_argument("--prefix", type=str, default="debug", help="Prefix is used mark datasets.")
    parser.add_argument("--weights", type=str, default="", help="Init weights path.weights/yolov5s.pt")


    parser.add_argument("--num_workers", type=int, default=16, help="Set num workers, for pytorch dataloader.")
    parser.add_argument("--device", type=int, default=0, help="You can set many devices.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank.")

    train_args = parser.parse_args()
    
    label_map = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",  "sheep",  "sofa",  "train", "tvmonitor"]
    train_args.label_map = label_map                            # 主要用来计算mAP、显示预测结果和真实标注的差距
    
    # visdom记录日志文件
    # vis = visdom.Visdom(env="yolov5s-debug")
    # train_args.vis = vis
    
    
    tensorboard_logs = LogRecoder(log_dir=os.path.join(train_args.save_dir, "logs"), flush_secs=10)
    train_args.tbd_logs = tensorboard_logs
    
    
    
    hyp = {
        "lr0": 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        "lrf": 0.15,  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": 0.937,  # SGD momentum/Adam beta1
        "weight_decay": 0.0001,  # optimizer weight decay 5e-4
        "warmup_epochs": 3.0,  # warmup epochs (fractions ok)
        "warmup_momentum": 0.8,  # warmup initial momentum
        "warmup_bias_lr": 0.1,  # warmup initial bias lr
        "sr": 0.001,            # sparse rate
        "box": 0.05,  # box loss gain
        "cls": 0.3,  # cls loss gain
        "cls_pw": 1.0,  # cls BCELoss positive_weight
        "obj": 0.7,  # obj loss gain (scale with pixels)
        "obj_pw": 1.0,  # obj BCELoss positive_weight
        "iou_t": 0.20,  # IoU training threshold
        "anchor_t": 4.0,  # anchor-multiple threshold
        # anchors: 3  # anchors per output layer (0 to ignore)
        "fl_gamma": 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
        "degrees": 0.0,  # image rotation (+/- deg)
        "translate": 0.1,  # image translation (+/- fraction)
        "scale": 0.9,  # image scale (+/- gain)
        "shear": 0.0,  # image shear (+/- deg)
        "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
        "flipud": 0.0,  # image flip up-down (probability)
        "fliplr": 0.5,  # image flip left-right (probability)
        "mosaic": 1.0,  # image mosaic (probability)
        "mixup": 0.1,  # image mixup (probability)
        "copy_paste": 0.1,  # segment copy-paste (probability)
    }
    
    


    # with open("/workspace/yolov5-pro/data/hyps/hyp.scratch-high.yaml", "r") as f:
    #     hyp = yaml.load(f, Loader=yaml.FullLoader)

    train(train_args, hyp)



































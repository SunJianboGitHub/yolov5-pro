#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plots.py
@Time    :   2022/09/28 14:31:59
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   plot tools
'''

import os
import cv2
import math
from copy import copy
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt





def feature_visualization(x, module_type, stage, max_n=32, save_dir="yolo_series/runs/detect/exp"):
    """
        ### 函数功能: detect阶段,显示各个网络层的特征
            x:                  需要被可视化的特征图
            module_type:        该网络层模块的名称
            stage:              当前网络层是整个模型的第几层
            max_n:              需要画出的特征图的最大数量,可以全部画出
            save_dir:           特征图绘制结果的保存路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if 'Detect' not in module_type:                                 # 只绘制卷积层输出的特征图，检测头并不绘制
        batch_size, channels, height, width = x.shape               # 解析特征图维度, NCHW
        if height > 1 and width > 1:                                # 如果特征图宽高都超过一个像素，才去绘制它
            save_file_name = save_dir + '/' + f"stage{stage}_{module_type.split('.')[-1]}_features.png"         # 当前层特征图的保存名称
            feature_blocks = torch.chunk(x[0].cpu().detach(), channels, dim=0)           # 这里只选取第一张图像的特征图，按照通道数进行拆分,返回的是列表，1*h*w
            max_n = min(max_n, channels)                                        # 实际绘制的通道数，防止设置太大，超过特征图通道数

            fig, ax = plt.subplots(math.ceil(max_n / 8), 8, tight_layout=True)
            ax = ax.ravel()                                                     # 将subplot的索引拉平，便于遍历
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(max_n):
                ax[i].imshow(feature_blocks[i].squeeze())                       # cmap='gray'
                ax[i].axis('off')                                                  # 关闭坐标轴
            
            plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
            plt.close()
            # np.save(str(save_file_name.replace('.png', '.npy')), x[0].cpu().detach().numpy())        # 特征图保存

 
def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=""):
    # 绘制学习率,模拟整个训练过程中的学习率变化
    optimizer, scheduler = copy(optimizer), copy(scheduler)
    y = []
    for _ in range(epochs):
        optimizer.step()
        scheduler.step()
        y.append(optimizer.param_groups[0]["lr"])

    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()
            


def xywh2xyxy_single(box):
    '''
        ### 函数功能: 单个边界框的坐标转换, xywh -> xyrb
            box 表示一个边界框, 属性是 x, y, w, h
    '''
    x, y, w, h = box
    left = x - (w - 1) / 2
    right = x + (w - 1) / 2
    top = y - (h - 1) / 2
    bottom = y + (h - 1) / 2
    return np.array([left, top, right, bottom])



def draw_bbox(image, bbox, confidence, class_id, color=(0, 255, 0), thickness=1):
    '''
        ### 函数功能: Detect时绘制检测边界框、类别、置信度
    '''
    left, top, right, bottom = [int(item + 0.5) for item in bbox]
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
    if class_id != -1:
        text = f"[{class_id}]{confidence:.2f}"
        cv2.putText(image, text, (left + 3, top - 5), 0, 0.5, (0, 0, 255), 1, 16)



def draw_norm_bboxes(image, norm_bboxes, color=(0, 255, 0), thickness=1):
    """
        ### 函数功能: 绘制yolo归一化后的边界框,验证数据增强后的边界框是否正确
            - 图像 image[ndarray]
            - bbox[Nx4/Nx5/Nx6]         框信息, 列数可以是4/5/6, 顺序是[cx, cy, width, height, confidence, classes], 基于图像大小进行归一化的框
    """
    img_height, img_width = image.shape[:2]                                         # 获取图像的宽高尺寸
    for norm_box in norm_bboxes:                                                    # 遍历每一个归一化的边界框
        box = norm_box * np.array([img_width, img_height] * 2)                      # 归一化边界框还原到图像尺寸
        box = xywh2xyxy_single(box)                                                 # xywh转换到xyrb
        
        confidence = 0                                                              # 主要针对检测评估阶段的结果显示, 目标置信度
        class_id = -1                                                               # 主要针对检测评估阶段的结果显示, 目标所属类别

        if len(norm_box) > 4:                                                       # 如果长度超过4, 索引4 表示目标置信度
            confidence = norm_box[4]
        if len(norm_box) > 5:                                                       # 如果长度超过5, 索引5 表示目标所属类别索引
            class_id = norm_box[5]
        draw_bbox(image, box, confidence, class_id, color, thickness)               # 在图像上绘制单个边界框


def draw_pixel_bboxes(image, pixel_bboxes, color=(0, 255, 0), thickness=1):
    """
        ### 函数功能: 绘制边界框,像素尺度,未归一化的边界框
            - 图像 image[ndarray]
            - bbox[Nx4/Nx5/Nx6]         框信息, 列数可以是4/5/6, 顺序是[left, top, right, bottom, confidence, classes], 基于图像大小进行归一化的框
    """
    for pixel_box in pixel_bboxes:
        left, top, right, bottom = pixel_box

        confidence = 0                                                                  # 主要针对检测评估阶段的结果显示, 目标置信度
        class_id = -1                                                                   # 主要针对检测评估阶段的结果显示, 目标所属类别

        if len(pixel_box) > 4:                                                          # 如果长度超过4, 索引4 表示目标置信度
            confidence = pixel_box[4]
        if len(pixel_box) > 5:                                                          # 如果长度超过5, 索引5 表示目标所属类别索引
            class_id = pixel_box[5]
        draw_bbox(image, pixel_box, confidence, class_id, color, thickness)             # 在图像上绘制单个边界框


def convert_norm_bboxs_to_pixel(norm_bboxes, img_width, img_height):
    """
        ### 函数功能: 将归一化的边界框复原
            - 转换标注信息, 从norm_bboxes 到 pixel_bboxes
            - norm_bboxes[Nx5]: 标注信息的格式为 [class_id, cx, cy, w, h]
            - img_width[int] : 标注信息的图像的宽度
            - img_height[int]: 标注信息的图像的高度
        ### 返回值
            - pixel_bboxes[Nx5]: 返回的格式是 [class_id, left, top, right, bottom]
    """
    pixel_bboxes = norm_bboxes.copy()
    class_id, cx, cy, width, height = [norm_bboxes[:, i] for i in range(5)]
    pixel_bboxes[:, 1] = cx * img_width - (width * img_width - 1) / 2               # left
    pixel_bboxes[:, 2] = cy * img_height - (height * img_height - 1) / 2            # top 
    pixel_bboxes[:, 3] = cx * img_width + (width * img_width - 1) / 2               # right 
    pixel_bboxes[:, 4] = cy * img_height + (height * img_height - 1) / 2            # bottom 
    
    return pixel_bboxes



def convert_pixel_bboxs_to_norm(pixel_bboxs, img_width, img_height):
    """
        ### 函数功能: 将边界框归一化,采用图像宽高进行归一化
            - 转换标注信息, 从pixel_bboxes 到 norm_bboxes
            - pixel_bboxes[Nx5]: 标注信息的格式为 [class_id, left, top, right, bottom]
            - img_width[int] : 标注信息的图像的宽度
            - img_height[int]: 标注信息的图像的高度
        ### 返回值
            - norm_bboxes[Nx5]: 返回的格式是 [class_id, cx, cy, w, h]
    """
    norm_bboxes= pixel_bboxs.copy()
    norm_bboxes[:, 1] = (pixel_bboxs[:, 1] + pixel_bboxs[:, 3]) * 0.5 / img_width          # cx
    norm_bboxes[:, 2] = (pixel_bboxs[:, 2] + pixel_bboxs[:, 4]) * 0.5 / img_height         # cy
    norm_bboxes[:, 3] = (pixel_bboxs[:, 3] - pixel_bboxs[:, 1] + 1) / img_width            # w
    norm_bboxes[:, 4] = (pixel_bboxs[:, 4] - pixel_bboxs[:, 2] + 1) / img_height           # h
    return norm_bboxes

















































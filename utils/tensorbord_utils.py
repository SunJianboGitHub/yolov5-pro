#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tensorbord_utils.py
@Time    :   2022/12/12 11:19:49
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   采用TensorBord 记录训练日志,并显示
'''

import os
import sys

from torch.utils.tensorboard import SummaryWriter


sys.path.append("..")

# 封装TensorBord接口,便于训练中日志记录


class LogRecoder:
    '''
        ### 功能说明
            1. 根据路径创建日志记录器
            2. 设置刷新频率,默认是120秒
            3. 可以根据方法,记录不同的日志.包括曲线、图像、分布、文本等
    
    '''
    def __init__(self, log_dir="run/exp1/results_1", flush_secs=10):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)             # 实例化一个可视化类对象, 日志记录器
        pass

    # 设置一条曲线,可以是损失函数、学习率、mAP、准确率等等
    def add_line(self, tag_name, scaler_value, global_step):
        self.writer.add_scalar(tag=tag_name,                            # 曲线的名称
                               scalar_value=scaler_value,               # 曲线纵坐标Y轴的值
                               global_step=global_step)                 # 曲线横坐标X轴的值

    # 绘制多条曲线,比如将多个mAP绘制在一张图上
    def add_lines(self, tag_name, tag_scalar_dict, global_step):
        self.writer.add_scalars(main_tag=tag_name,                      # 图像的总名称
                                tag_scalar_dict=tag_scalar_dict,        # 每条曲线表示的内容
                                global_step=global_step)                # 对应的步长
        
    # 绘制图像,可以用来显示增强后的图像,或者检测结果图
    def add_image(self, tag_name, img, global_step, walltime, dataformats="CHW"):
        self.writer.add_image(tag=tag_name,                                 # 图像的名称
                              img_tensor=img,                           # 图像数值数据
                              global_step=global_step,                  # 步长
                              walltime=walltime,
                              dataformats=dataformats)                  # 图像格式,默认应该是CHW
    
    # 绘制多张图像,可用来显示特征图
    def add_images(self, name, img, global_step, dataformats="NCHW"):
        self.writer.add_images(tag=name,                                # 图像的名称
                               img_tensor=img,                          # 图像的数值数据
                               global_step=global_step,                 # 步长 
                               dataformats=dataformats)                 # 图像的格式,默认应该是NCHW
    
    
    # 显示figure对象到TensorBord的网页端,用于展示一些较为复杂的图片
    def add_figure(self, name, fig, global_step):
        self.writer.add_figure(tag=name,                                # 图片的名称
                               figure=fig,                              # 需要显示的图片
                               global_step=global_step)                 # 步长
        
    # 显示频率分布直方图,主要用于显示权重和梯度分布
    # 比如,通道剪枝时用来显示BN权重分布
    def add_histogram(self, tag_name, values, global_step, bins="tensorflow"):
        self.writer.add_histogram(tag=tag_name, values=values, global_step=global_step, bins=bins)              # 当前值直方图对应的步长
                                  
    
    # 添加文本信息
    # 增加文本训练日志
    def add_text(self, tag_name, text_str, global_step):
        self.writer.add_text(tag=tag_name,                              # 文件的名称
                             text_string=text_str,                      # 写入的字符串
                             global_step=global_step)                   # 步长
        
        pass
    

    def __del__(self):
        
        self.writer.close()





if __name__ == "__main__":
    
    import time
    import cv2
    tbd_logs = LogRecoder("/workspace/yolov5-pro/runs/train/exp1/logs", flush_secs=1)
    
    image = cv2.imread("../test_show.jpg")

    epochs = 2000
    for epoch in range(1000, epochs):
        
        loss_1 = 1 - 0.9 * (epoch / epochs)
        loss_2 = 1 - 0.3 * (epoch / epochs)
        
        tbd_logs.add_line("loss/train_loss", loss_1, epoch)
        tbd_logs.add_line("loss/val_loss", loss_2, epoch)
        
        tbd_logs.add_text("loss/logger_train_loss", str(loss_1), epoch)
        
        tbd_logs.add_image("image/show_image", image, epoch % 3, "HWC")
        
        
        
        time.sleep(0.1)
        print(epoch)




































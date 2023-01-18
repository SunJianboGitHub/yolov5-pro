#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   detect.py
@Time    :   2023/01/10 11:40:21
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   yolov5 detection
'''
import os
import cv2
import sys
import time
import torch
import torchvision
import argparse
import numpy as np

from models.yolo import YoloV5
from utils.torch_utils import non_max_suppression

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


# 加载模型训练权重, 并设置为推理模式
def load_model_state_dict(model, opts):
    ckpt = torch.load(opts.weights)
    model.load_state_dict(ckpt['model'])
    model.to(opts.device)
    model.eval()


# 图像预处理
def image_preprocess(img_path, img_size=640):
    ori_image = cv2.imread(img_path)
    h, w, _ = ori_image.shape
    scale = min(img_size / h, img_size / w)
    
    offset_x, offset_y = (img_size/2 - w*scale/2), (img_size/2 - h*scale/2)                # 仿射变换的偏移量
    M = np.array([
        [scale, 0, offset_x],
        [0, scale, offset_y]
    ], dtype=np.float32)
    
    input_image = cv2.warpAffine(ori_image, M, (img_size, img_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

    # cv2.imwrite("test_show_2.jpg", input_image)
    # cv2.imwrite("test_show_3.jpg", ori_image)
    
    return ori_image, input_image, M


# 结果后处理
def result_postprocess(predictions, M):
    invert_M = cv2.invertAffineTransform(M)
    for i, predict in enumerate(predictions):
        if predict.shape[0] != 0:
            one_tensor = torch.ones((predict.shape[0], 1), device=predict.device)
            xy = (torch.cat((predict[:, :2], one_tensor), dim=1) @ torch.from_numpy(invert_M).cuda().T)
            rb = (torch.cat((predict[:, 2:4], one_tensor), dim=1) @ torch.from_numpy(invert_M).cuda().T)

            predict[:, :2] = xy
            predict[:, 2:4] = rb
        
    return predictions


# 图像转换为torch tensor
def image_to_tensor(input_img, opts, std=True, data_type=None):
    input_img = np.ascontiguousarray(input_img.transpose(2, 0, 1)[::-1])             # 先转换为CHW, 再转换为 RGB, 然后再连续化
    input_tensor = torch.from_numpy(input_img)[None]
    input_tensor = input_tensor.to(opts.device).float() / 255
    if std and data_type:
        normalize = acquire_mean_std(data_type=data_type)
        input_tensor = torchvision.transforms.Normalize(mean=normalize[0], std=normalize[1])(input_tensor)
    
    return input_tensor


# 绘制边界框
def draw_bboxes(image, predicts):
    for predict in predicts:
        x, y, r, b = [int(item) for item in predict[:4]]
        score, cls = predict[4:]
        txt = str("p = ") + str(score)[:5] + " cls = " +  str(int(cls))
        cv2.rectangle(image, (x, y), (r, b), (255, 0, 0), 2)
        cv2.putText(image, txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 255), thickness=1)



def detect(opts):
    model = YoloV5(yaml_cfg_file=opts.cfg)
    load_model_state_dict(model, opts)
    
    imgs_list = os.listdir(opts.data)
    
    time_list = []
    
    for img_name in imgs_list:
        
        # 统计时间：前处理-推理-后处理
        start = time.time()
        
        # 图像读取与预处理
        img_path = os.path.join(opts.data, img_name)
        ori_img, input_img, transform_M = image_preprocess(img_path=img_path, img_size=opts.img_size)
        
        # 这里得到的预测结果是在输入尺度图像上的预测结果
        input_tensor = image_to_tensor(input_img, opts, std=opts.img_std, data_type=opts.norm_mode)
        predictions = model(input_tensor)
        predictions = non_max_suppression(predictions, iou_thres=opts.iou_thres, conf_thres=opts.conf_thres, max_output_det=opts.max_det)
        
        
        # 后处理, 恢复到原始图像上的box
        predictions = result_postprocess(predictions, transform_M)[0]

        time_list.append(time.time() - start)
        
        # 在原始图像上绘制边界框
        predictions = predictions.detach().cpu().numpy()
        draw_bboxes(ori_img, predictions)
        
        # 保存图像到本地
        save_name = os.path.join(opts.save_dir, img_name)
        cv2.imwrite(save_name, ori_img)
        
    
    print("Inference time: ", np.mean(np.array(time_list)) * 1000, " ms.")

    
    return 0







if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="/workspace/yolov5-pro/models/yolov5s.yaml", help="Network yaml config file.")
    parser.add_argument("--data", type=str, default="/workspace/yolov5-pro/data/excavator_images", help="Input Imags dir.")
    parser.add_argument("--save_dir", type=str, default="/workspace/yolov5-pro/data/results", help="Save detect results dir.")
    parser.add_argument("--iou_thres", type=float, default=0.50, help="NMS IoU threshold.")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Confidence score.")
    parser.add_argument("--max_det", type=int, default=100, help="Max output boxes.")
    
    parser.add_argument("--img_std", type=bool, default=True, help="Image normalize.")
    parser.add_argument("--norm_mode", type=str, default="coco", help="Mean and std from datasets type.")
    
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=int, default=0, help="You can set many devices.")
    parser.add_argument("--weights", type=str, default="/workspace/yolov5-pro/runs/train/excavator2023-exp1/weights/best.pt", help="yolov5s.pt")
    
    opts = parser.parse_args()
    
    detect(opts)
    
    pass































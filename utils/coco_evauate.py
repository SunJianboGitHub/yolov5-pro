#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   coco_evaluate.py
@Time    :   2022/12/05 10:04:44
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   None
'''

import os
import cv2
import sys
import json
import torch
import numpy as np
import torchvision
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


sys.path.append("..")

from utils.dataset import create_dataloader
from utils.yolo2coco import yolo2coco


# 利用COCO API 计算mAP
class COCOmAP:
    def __init__(self, show=True):
        self.show = show
        
    
    

    def apply(self, json_gt_anno_file, pt_anno_coco_json):
        
        # 表示未检测看到任何目标
        if len(pt_anno_coco_json) == 0:
            print("Model can not detect object, pt_anno_coco_json = [], So do not evaluate...")
            return 0, 0, 0
        
        cocoGt = COCO(json_gt_anno_file)
        cocoDt = cocoGt.loadRes(pt_anno_coco_json)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP, AP50, AP75 = cocoEval.stats[:3]
        return mAP, AP50, AP75
        


def recover_bbox(img_size, ori_h, ori_w, pt_bboxes):
    """
        ### 函数说明
            1. img_size, 表示推理时的图像大小
            2. ori_h, 表示原始图像的高度
            3. ori_w, 表示原始图像的宽度
            4. pt_bboxes, 表示在img_size尺度下的检测框, 维度是[M, 6], [left, top, right, bottom, score, clas_id]
    """
    scale = min(img_size / ori_w, img_size / ori_h)
    offset_x, offset_y = (img_size/2 - (ori_w*scale)/2), (img_size/2 - (ori_h*scale)/2)
    pt_bboxes[:, :4] -= torch.tensor([offset_x, offset_y, offset_x, offset_y], device=pt_bboxes.device)
    pt_bboxes[:, :4] *= torch.tensor([1/scale, 1/scale, 1/scale, 1/scale], device=pt_bboxes.device)
    
    pt_bboxes[:, 0].clamp_(0, ori_w)  # x1
    pt_bboxes[:, 1].clamp_(0, ori_h)  # y1
    pt_bboxes[:, 2].clamp_(0, ori_w)  # x2
    pt_bboxes[:, 3].clamp_(0, ori_h)  # y2
    
    return pt_bboxes


def xywh2xyrb(bboxes):
    """
        ### 函数说明
            - 转换多个边界框的坐标
            - bboxes: [N, 4], [x, y, w, h]
    """
    left   = bboxes[:, 0] - (bboxes[:, 2] - 1) * 0.5
    right  = bboxes[:, 0] + (bboxes[:, 2] - 1) * 0.5
    top    = bboxes[:, 1] - (bboxes[:, 3] - 1) * 0.5
    bottom = bboxes[:, 1] + (bboxes[:, 3] - 1) * 0.5
    bboxes[:, 0] = left
    bboxes[:, 1] = top
    bboxes[:, 2] = right
    bboxes[:, 3] = bottom
    return bboxes


# batched NMS, 对多张图像的推理结果,执行非极大值抑制,得到预测框
def non_max_suppression(predictions, iou_thres=0.5, conf_thres=0.001, max_output_det=30000):
    """
        ### 参数
            - predictions: [[N, 5+20], [N, 5+20], ...], [x, y, w, h, conf, cls_p1, cls_p2...]
            - iou_thres: 计算mAP时, 应该设置为0.5
            - conf_thres: 计算mAP时, 应该设置为0.001
        ### 执行过程说明
            - 根据conf_thresh筛选预测框, 计算mAP会设置为0.001, 在detect时, 会设置的大一些,比如0.2-0.5
            - yolov5采用conf_thresh筛选2次, 一次是objectnes > conf_thres, 一次是 confidence_score > conf_thres
            - yolov5中提到的confidence_score是指 objectness * classification
            - 我觉得筛选两次和一次没什么区别,主要保证 score > conf_thres
        ### 返回值
            - output是一个list
            - list中的每一个元素就是一个图像的检测框
            - 检测框的返回结构是[left, top, right, bottom, conf_score, class_id]
    """
    # 检查输入参数
    assert isinstance(predictions, torch.Tensor), f'Predictions Tensor dims must: [bs, num_box, 5+cls]'
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are [0, 1].'
    assert 0 <= iou_thres <= 1, f'Invalid IoU threshold {iou_thres}, valid values are [0, 1].'

    max_wh = 7680                                                                # box 宽度和高度的最大值
    max_nms = 30000                                                              # torchvision.ops.nms()处理的最大box数量

    bs = predictions.shape[0]                                                    # 批量大小
    output = [torch.zeros((0, 6), device=predictions.device)] * bs               # 输出列表,每个元素就是一张图像的预测结果

    # 初步筛选,这一步可以不做,这里主要减少候选框数量,可提速
    # objectness > conf_thres
    select_mask = predictions[..., 4] > conf_thres                  # 维度是[bs, num_boxes]=[bs, 25200]
    
    for i, predict in enumerate(predictions):                       # 遍历每一张图像的预测结果
        predict = predict[select_mask[i]]                           # 选出 objectness > conf_thres部分检测结果, 维度是 [N, 5+cls]
        bbox = xywh2xyrb(predict[:, :4])                            # 转换边界框, 从 xywh->left,top,right,bottom, clamp(0, 640-1)?
        predict[:, 5:] *= predict[:, 4:5]                           # 计算score = objectness * classification
        score, cls_id = predict[:, 5:].max(1, keepdim=True)
        predict = torch.cat((bbox, score, cls_id.float()), dim=1)[score.view(-1) > conf_thres]
        
        num_box = predict.shape[0]
        
        if not num_box:                                 # 没有检测框,直接跳过
            continue
        elif num_box > max_nms:                         # 超过torchvision.ops.nms()处理的最大box数量, 排序之后取前面部分
            predict = predict[predict[:, 4].argsort(descending=True)[:max_nms]]
        else:
            predict = predict[predict[:, 4].argsort(descending=True)]

        bboxes, scores, cls_ids = predict[:, :4], predict[:, 4], predict[:, 5]
        keep_idx = torchvision.ops.batched_nms(bboxes, scores, cls_ids, iou_thres)
        if keep_idx.shape[0] > max_output_det:                                       #只输出 一张图像的最大预测数目的边界框,和计算mAP还不太一样
            keep_idx = keep_idx[:max_output_det]
        
        output[i] = predict[keep_idx]

    return output


def generate_predicts_coco_json(model, val_img_txt, img_map_id, prefix="debug", img_size=640, batch_size=16, num_workers=16,
                                   max_det=30000, nms_thres=0.5, conf_thres=0.001, device="cuda"):
    """
        ### 函数说明
            - 根据训练的模型对验证集数据进行推理,并将结果保存,用于计算mAP
            - 推理的结果应该先通过 conf_thres筛选,然后再进行类内的NMS. 最后将结果变换为想要的格式[left,top,right,bottom,conf_score,cls_id]
            - 最后,将每个图像的名称作为键与其检测框匹配,并将所有的图像与其检测结果存储到dict中,并保存起来(非必须)
    """
    data_loader, data_sets = create_dataloader(datas_path=val_img_txt,                                # 数据集的路径
                                         hyp=None,                                                    # 超参数，暂时没用到
                                         shuffle=False,                                              # 打乱数据集，训练集打乱，测试集不打乱                                 
                                         augment=False,                                      # 基础的图像增强
                                         mixed_aug=False,                                  # 马赛克增强和中心缩放
                                         cache_images=True,                            # 是否缓存图像，根据数据集大小确定是否缓存
                                         mosaic_nums=[4],                               # 马赛克增强的方法，也就是多少个图片进行镶嵌增强
                                         prefix=prefix,                                        # 前缀，用于区分数据集
                                         batch_size=batch_size,                                # 网络的输入批次大小
                                         img_size=img_size,                                    # 网络的输入尺寸
                                         num_workers=num_workers,                              # 读取数据集，开启的线程数
                                         border_fill_value=114)                                     # 图像边界的填充数值
    
    model = model.to(device)
    model.eval()
    
    anno_dets = []
    
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_sets)//batch_size, desc="Compute mAP...")
        for i, (images, labels, visual_info) in pbar:
            images, labels = images.to(device), labels.to(device)
            predicts = model(images).detach()
            predicts = non_max_suppression(predicts, nms_thres, conf_thres, max_det)        # 对预测结果进行batched_nms,                       
            for j in range(len(predicts)):
                index = i * batch_size + j                                                  # 计算计算到了第几个图像用于将图像名称与预测框对应    
                idx = data_sets.image_files[index].rfind("/")                               # 图像的最后一个/的索引
                ori_h, ori_w = cv2.imread(data_sets.image_files[index]).shape[:2]           # 图像的原始宽高
                
                img_name = data_sets.image_files[index][idx+1:]                             # 图像名称
                # print("img_name: ", img_name)
                img_id = img_map_id[img_name]                                               # 图像ID
                pt_bboxes = predicts[j]                                                     # 维度是[M, 6], [left, top, right, bottom, score, clas_id]
                pt_bboxes = recover_bbox(img_size, ori_h=ori_h, ori_w=ori_w, pt_bboxes=pt_bboxes)
                
                for left, top, right, bottom, score, cls_id in pt_bboxes:
                    width = right - left + 1
                    height = bottom - top + 1
                    object_item = {"image_id": img_id, "category_id": cls_id, "score": score, "bbox":[left, top, width, height]}
                    anno_dets.append(object_item)
                               
        return anno_dets, len(anno_dets)



# 根据训练的模型，评估数据集，计算mAP
def estimate(model, val_img_txt, image_size=640, batch_size=16, num_workers=16,
                                   nms_max_output_det=30000, nms_thres=0.5, conf_thres=0.001, device="cuda"):
    """_summary_

    参数说明:
        model (_type_): 网络模型
        val_img_txt (_type_): 验证集文件路径
        image_size (int, optional): 评估验证模型时输入分辨率. Defaults to 640.
        batch_size (int, optional): 评估推理时批的大小. Defaults to 16.
        num_workers (int, optional): 线程数目. Defaults to 16.
        nms_max_output_det (int, optional): NMS后保留的最大框数目. Defaults to 30000.
        map_max_det (int, optional): 计算mAP时通常选取前100个置信度最大的框(在每张图像上). Defaults to 100.
        nms_thres (float, optional): NMS的阈值. Defaults to 0.5.
        conf_thres (float, optional): 计算mAP时通常选取较小的置信度阈值,确保较高的召回率. Defaults to 0.001.
        device (str, optional): 评估时,选择推理设备. Defaults to "cuda".
    """
    # print("Starting evalate validation dataset...")
    model.eval()
    
    root_dir = Path(val_img_txt).parent                                                         # 获取验证集的根目录
    prefix = (val_img_txt[val_img_txt.rfind("/")+1:]).split(".")[0]                              # 方便区分数据  

    json_gt_anno_file =str(root_dir) + os.sep + "caches" + os.sep + prefix +"_gt_anno_coco.json"
    json_img_map_id = str(root_dir) + os.sep + "caches" + os.sep + prefix + "_img_map_id.json"

    
    # 检查缓存路径是否存在,不存在的话,创建它
    if not os.path.exists(os.path.join(root_dir, "caches")):
        os.makedirs(os.path.join(root_dir, "caches"))

    if os.path.exists(json_gt_anno_file) and os.path.exists(json_img_map_id):
        with open(json_gt_anno_file, "r") as fr:
            gt_anno_coco_json = json.load(fr)
        with open(json_img_map_id, "r") as fr:
            img_map_id = json.load(fr)
            num_gt = img_map_id["num_gt"]
    else:
        gt_anno_coco_json,  img_map_id, num_gt= yolo2coco(val_img_txt, prefix=prefix)
        # 保存COCO格式的标签信息
        with open(json_gt_anno_file, "w") as fw:
            json.dump(gt_anno_coco_json, fw, indent=4, ensure_ascii=False)

        # 保存图像名称和ID的映射信息
        img_map_id["num_gt"] = num_gt
        with open(json_img_map_id, "w") as fw:
            json.dump(img_map_id, fw, indent=4, ensure_ascii=False)
    
    # print(img_map_id)
    
    # 预测验证集,并将结果整理成COCO的格式
    pt_anno_coco_json, num_pt = generate_predicts_coco_json(model, val_img_txt, img_map_id, prefix, 
                                                            img_size=image_size,
                                                            batch_size=batch_size,
                                                            num_workers=num_workers,
                                                            max_det=nms_max_output_det,
                                                            nms_thres=nms_thres,
                                                            conf_thres=conf_thres,
                                                            device=device)
    # with open(json_pt_anno_file, "w") as fw:
    #     json.dump(pt_anno_coco_json, fw, indent=4, ensure_ascii=False)
    
    print("num_pt: ", num_pt)
    print("num_gt: ", num_gt)
    
    # 使用COCO API 开始计算mAP
    mAP, AP50, AP75 = COCOmAP(show=True).apply(json_gt_anno_file, pt_anno_coco_json)
    
    return mAP, AP50, AP75
        
    # print(mAP_list)
    # print("mAP@0.50     = {:.3f}".format(mAP_list[0]))
    # print("mAP@0.75     = {:.3f}".format(mAP_list[5]))
    # print("mAP@0.5:0.95 = {:.3f}".format(np.mean(np.array(mAP_list))))
    
    
    # return mAP_list[0], mAP_list[5], np.mean(np.array(mAP_list))




if __name__ == "__main__":
    import cv2
    
    gt_anno_file = "/workspace/datasets/PASCAL_VOC2007/VOC2007_trainval/caches/debug_gt_anno_coco.json"
    with open(gt_anno_file, "r") as fr:
        gt_anno_json = json.load(fr)
        
    images = gt_anno_json["images"]
    annotations = gt_anno_json["annotations"]
    
    for i in range(len(images)):
        img_path = images[i]["file_name"]
        img_id = images[i]["id"]
        image = cv2.imread(img_path)
        if i == 2:
            for anno in annotations:
                if anno["image_id"] == img_id:
                    x, y, w, h = [int(item) for item in anno["bbox"]]
                    r, b = x + w, y + h
                    cv2.rectangle(image, (x, y), (r, b), (255, 0, 0), 3)
            
            cv2.imwrite("kkkkkkkkkkkkkkkkkk.jpg", image)
            break
    
    # print(gt_anno_json)


















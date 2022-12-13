#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   yolo2coco.py
@Time    :   2022/12/05 12:29:05
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   None
'''

import os 
import cv2
import json
from pathlib import Path


# 当采用 COCO API 计算mAP时,需要将groundtruth标签转换为 Json格式
# 下面将根据验证集、测试集的txt文件,生成COCO格式的json标注文件

VOC_NAMES = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",  "sheep",  "sofa",  "train", "tvmonitor"]




# COCO.json中保存的每一个图像信息
class one_img_dict:
    def __init__(self):
        self.id, self.width, self.height = [-1] * 3
        self.license = 0
        self.file_name, self.flickr_url, self.color_url, self.date_captured = [""] * 4
    
    def dict_info(self):
        assert self.id >= 0 and self.width >= 0 and self.height >= 0 and self.file_name != "",  "Please input correctly infos..."
        return {
            "id": self.id, "width": self.width, "height": self.height, "license": self.license,
            "file_name": self.file_name, "flickr_url": self.flickr_url, "color_url": self.color_url, "date_captured": self.date_captured
        }


# COCO.json中保存的每一个标注信息
class one_anno_dict:
    def __init__(self):
        self.ann_id, self.image_id, self.category_id, self.area = [-1] * 4
        self.segmentation = [[]]
        self.bbox = []
        self.iscrowd = 0
        self.attributes = ""
    
    def dict_info(self):
        assert self.ann_id >= 0 and self.image_id >= 0 and self.category_id >= 0 and self.area >= 0,  "Please input correctly infos..."
        return {
            "id": self.ann_id,  "image_id": self.image_id, "category_id": self.category_id, "area": self.area,
            "segmentation": self.segmentation,
            "bbox": self.bbox,
            "iscrowd": self.iscrowd,
            "attributes": self.attributes
        }



def yolo2coco(images_txt, prefix="voc2007_val"):
    
    root_dir = Path(images_txt).parent                              # 获取验证集的根目录
    
    images = []                                                     # 存储所有的图像
    annotations = []                                                # 存储所有的标注信息
    categories = []                                                 # 存储所有类别信息
    img_map_id = dict()                                             # 图像的名称可能不是数字,但是COCO中的图像ID必须是数字,因此在这里做映射,使得名称与ID一一对应
    
    for label_name in VOC_NAMES:
        categories.append({"supercategory": label_name, "name": label_name, "id": VOC_NAMES.index(label_name)})
    
    # print(categories)
    num_gt = 0                      # 标注的目标数目
    img_id = 1000000                # 图像的标号
    ann_id = 0                      # 标注框的标号
    
    # 读取验证集图像列表
    with open(images_txt, "r") as f:
        img_list = f.readlines()
        img_list = [item.strip() for item in img_list]              # 去除行尾的换行符
    
    # 遍历验证集列表,整理图像和标注信息
    for img_path in img_list:
        img_h, img_w = cv2.imread(os.path.join(root_dir, img_path)).shape[:2]       # 获取图像的宽高
        
        # 添加图像名称和图像ID的映射,保证一一对应
        img_name = img_path[img_path.rfind("/")+1:]                          # 获取图像的名称
        img_map_id[img_name] = img_id                                       # 映射图像名称到ID
        
        # 添加图像属性
        img_info = one_img_dict()
        img_info.id = img_id
        img_info.width = img_w
        img_info.height = img_h
        img_info.file_name = os.path.join(root_dir, img_path)
        images.append(img_info.dict_info())
        
        # 读取当前图像对应的标注文件
        lab_path = os.path.join(root_dir, img_path.replace("JPEGImages", "labels").replace(".jpg", ".txt"))
        with open(lab_path, "r") as f:
            lab_list = f.readlines()
            lab_list = [item.strip() for item in lab_list]
        
        # 添加anno标注信息
        for line in lab_list:
            ann_id += 1                                                                         # 标注的id从1开始
            cls_id, x, y, w, h = line.strip().split(" ")                                        # 获取每一个标注框的详细信息
            cls_id, x, y, w, h = int(cls_id), float(x), float(y), float(w), float(h)            # 将字符串类型转为可计算的int和float类型

            # 坐标转换
            xmin = (x - (w-1) / 2) * img_w
            ymin = (y - (h-1) / 2) * img_h
            xmax = (x + (w-1) / 2) * img_w
            ymax = (y + (h-1) / 2) * img_h
            w = w * img_w
            h = h * img_h
            
            # 添加一个标注信息
            lab_info = one_anno_dict()                                                          # 将每一个bounding box信息存储在该字典中
            lab_info.ann_id = ann_id                                                            # 每个标注信息的索引
            lab_info.image_id = img_id                                                          # 当前图像的ID索引
            lab_info.category_id = cls_id                                                       # 当前标注的类别信息
            lab_info.segmentation = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]          # 分割信息
            lab_info.area = w * h                                                               # 边界框的面积
            lab_info.bbox = [xmin, ymin, w, h]                                                  # 边界框坐标,左上角以及宽高
            annotations.append(lab_info.dict_info())                                            # 添加标签信息到列表
            
            num_gt += 1                                                                         # 累加计算标注的目标数目
    
        img_id += 1                                                                             # 递增图像ID
    
    # 准备写出信息
    write_json_dict = dict()
    write_json_dict["images"]      = images
    write_json_dict["annotations"] = annotations
    write_json_dict["categories"]  = categories
    write_json_dict["licenses"]    = [{'name': "", 'id': 0, 'url': ""}]
    write_json_dict["info"]        = {'contributor': "", 'date_created': "", 'description': "", 'url': "", 'version': "", 'year': ""}

    
    return write_json_dict, img_map_id, num_gt
    
    # # 检查缓存路径是否存在,不存在的话,创建它
    # if not os.path.exists(os.path.join(root_dir, "caches")):
    #     os.makedirs(os.path.join(root_dir, "caches"))

    # # 保存COCO格式的标签信息
    # json_file = os.path.join(root_dir, "caches", prefix +"_gt_anno_coco.json")
    # with open(json_file, "w") as fw:
    #     json.dump(write_json_dict, fw, indent=4, ensure_ascii=False)

    # # 保存图像名称和ID的映射信息
    # json_img_map_id = os.path.join(root_dir, "caches", prefix + "_img_map_id.json")
    # with open(json_img_map_id, "w") as fw:
    #     json.dump(img_map_id, fw, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    images_txt = "/workspace/datasets/PASCAL_VOC2007/VOC2007_trainval/debug.txt"
    yolo2coco(images_txt, "debug")























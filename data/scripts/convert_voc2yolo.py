#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   convert_voc2yolo.py
@Time    :   2022/09/30 21:16:32
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   将VOC数据的xml格式数据转换成YOLO的txt格式
'''

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm



# 边界框的坐标转换
def xyrb2xywh(box):
    """
        box: 它是一个list, 没有进行归一化
    """
    box = [float(item) for item in box]
    left, top, right, bottom = box
    x = (left + right) / 2.0
    y = (top + bottom) / 2.0
    w = (right - left) + 1.0
    h = (bottom - top) + 1.0
    return [x, y, w, h]


# 解析voc数据集中Annotations文件夹下的xml文件
def parse_xml(xml_name):
    with open(xml_name, "r") as xml_file:
        tree = ET.parse(xml_file)
        root_node = tree.getroot()
        size = root_node.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)
        
        label_infos = []
        for object in root_node.iter("object"):
            name = object.find("name").text
            bbox = object.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            bbox = xyrb2xywh([xmin, ymin, xmax, ymax])
            bbox = list(np.array(bbox) * np.array([1./img_w, 1./img_h] * 2))
            bbox.insert(0, name)
            label_infos.append(bbox)
        return label_infos


def write_label_infos_to_txt(label_infos, classes_name, txt_path, txt_name):
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    with open(os.path.join(txt_path, txt_name), "w") as f:
        for i in range(len(label_infos)):
            label_info = label_infos[i]
            cls_id = classes_name.index(label_info[0])
            f.write(str(cls_id) + " ")
            for i in range(4):
                f.write(str(label_info[i+1]) + " ")
            f.write("\n")




if __name__ == "__main__":
    root_path = "/workspace/CourseCode/datasets/classic_datasets/PASCAL_VOC2012"
    classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # part_dir = ["VOC2007_test", "VOC2007_trainval"]

    num_list = []
    pbar = tqdm(os.listdir(root_path))
    for dir in pbar:
        xml_path = os.path.join(root_path, dir, "Annotations")
        labels_path = xml_path.replace("Annotations", "labels")
        num = 0
        for xml_name in os.listdir(xml_path):
            if not xml_name.endswith(".xml"):
                continue
            txt_path = xml_path.replace("Annotations", "labels")
            label_infos = parse_xml(os.path.join(xml_path, xml_name))
            txt_name = xml_name.replace(".xml", ".txt")
            write_label_infos_to_txt(label_infos, classes_name, txt_path, txt_name)

            num += 1

        num_list.append(num)



    print(f"num_list = {num_list[0]}, {num_list[1]}")












#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   convert_ccpd2yolo.py
@Time    :   2022/09/30 14:40:03
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   ccpd数据集转换成YOLO格式 分割数据集为训练、验证、测试
'''


import os 
import cv2
import numpy as np
from tqdm import tqdm

'''
    x = (left + right) / 2.0
    y = (top + bottom) / 2.0
    width = (right - left) + 1.0
    height = (bottom - top) + 1.0

    left = x - (width - 1) / 2.0
    right = x + (width - 1) / 2.0
    top = y - (heigth - 1) / 2.0
    right = y + (height - 1) / 2.0
'''

def xywh2xyrb(box):
    """
        box: 它是一个np.array格式 未进行归一化的数据
    """
    x, y, w, h = box

    left = x - (w - 1) * 0.5
    right = x + (w - 1) * 0.5
    top = y - (h - 1) * 0.5
    bottom = y + (h - 1) * 0.5
    return np.array([left, top, right, bottom])


def xyrb2xywh(box):
    """
        box: 它是一个np.array格式 未进行归一化的数据
    """
    box = np.array([float(item) for item in box])
    left, top, right, bottom = box
    x = (left + right) / 2.0
    y = (top + bottom) / 2.0
    width = (right - left) + 1.0
    height = (bottom - top) + 1.0
    return np.array([x, y, width, height])



# 给定一个根目录，读取该目录下的所有文件
def get_all_files_base_rootdir(root, res):
    try:
        allfiles = os.listdir(root)
        for eachfile in allfiles:
            if eachfile.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                res.append(os.path.join(root, eachfile))
            else:
                new_path = os.path.join(root, eachfile)
                get_all_files_base_rootdir(new_path, res)
    except Exception as e:
        print(e)

# 解析图像名字坐标点的函数
def split_point(str_anno):
    result = []
    pair_points = str_anno.split('_')
    for point in pair_points:
        coord = point.split('&')
        result.extend(coord)
    return result


# 根据图像名称获取标注信息, 这里返回值是归一化后的结果
def acquire_info_base_image(img_path):
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    img_name = img_path.split('/')[-1]
    name_list = img_name.split("-", 4)             # 以 - 分割, 只分割出4部分，也就是前三个“-”分割的部分
    if len(name_list) != 5:
        return []
    
    # 位置0-表示为区域
    # 位置1-表示为水平角度和垂直角度
    # 位置2-表示为对应边界框的左上角和右下角坐标
    # 位置3-表示车牌对应的4个顶点

    bbox = split_point(name_list[2])
    landmarks = split_point(name_list[3])

    # 归一化
    dw = 1./w
    dh = 1./h
    box = np.array([float(item) for item in bbox])
    landmarks = np.array([float(item) for item in landmarks])
    norm_landm = np.array([dw, dh] * 4)
    norm_box = np.array([dw, dh] * 2)
    box = xyrb2xywh(box) * norm_box
    landmarks = landmarks * norm_landm
    return box, landmarks


# 将归一化后的标签信息写入指定文件
def write_label_info_to_txt(img_file, norm_box, norm_landmarks):
    label_file = img_file.replace('images/', 'labels/').replace('.jpg', '.txt')
    last_dir = label_file[:label_file.rfind("/")]
    if not os.path.exists(last_dir):
        os.makedirs(last_dir)
    with open(label_file, "w") as f:
        f.write("0 ")
        for i in range(len(norm_box)):
            f.write(str(norm_box[i]) + " ")
        for i in range(len(norm_landmarks)):
            f.write(str(norm_landmarks[i]) + " ")
        f.write('\n')


def write_img_list_to_txt(img_list, txt_file):
    with open(txt_file, "w") as f:
        for i in range(len(img_list)):
            f.write(img_list[i] + "\n")




def split_datasets_trainval_test(all_imgs_list, num_val = 5000, num_test = 10000, save_dir="./"):
    np.random.seed(3)
    np.random.shuffle(all_imgs_list)

    val_txt = os.path.join(save_dir, "val.txt")
    test_txt = os.path.join(save_dir, "test.txt")
    train_txt = os.path.join(save_dir, "train.txt")
    trainval_txt = os.path.join(save_dir, "trainval.txt")

    test_list = all_imgs_list[:num_test]
    val_list = all_imgs_list[num_test: num_test+num_val]
    train_list = all_imgs_list[num_test+num_val:]
    trainval_list = all_imgs_list[num_test:]

    write_img_list_to_txt(val_list, val_txt)
    write_img_list_to_txt(test_list, test_txt)
    write_img_list_to_txt(train_list, train_txt)
    write_img_list_to_txt(trainval_list, trainval_txt)

    return (len(test_list), len(val_list), len(train_list), len(trainval_list))


if __name__ == "__main__":
    root_path = "/workspace/CourseCode/datasets/ccpd_datasets"
    img_dir = os.path.join(root_path, "images")

    all_imgs_path = []
    get_all_files_base_rootdir(img_dir, all_imgs_path)                  #遍历多级目录 得到所有图像的绝对路径，ccpd数据集 一共包含 366786 个图像

    error_examples = 0
    all_selected_imgs = []
    pbar = tqdm(range(len(all_imgs_path)))
   
    for i in pbar:
        try:
            img_file = all_imgs_path[i]                                             # 获取每一张图像的绝对路径
            img_part_name = img_file.split("ccpd_datasets/")[-1]
            if os.path.exists(img_file.replace("/images", "/labels").replace(".jpg", ".txt")):
                all_selected_imgs.append(img_part_name)
                continue
            label_info = acquire_info_base_image(img_file)            # 根据图像名称, 获取每一个图像的标注信息，已经归一化了

            if label_info == []:                                      # 表示当前图像的标注信息有问题, 这张图像直接丢弃，不会用来训练、测试
                error_examples += 1
                continue

            norm_box, norm_landmarks = label_info[0], label_info[1]
            write_label_info_to_txt(img_file, norm_box, norm_landmarks)         # 将标注信息写入指定文件

            all_selected_imgs.append(img_part_name)
        except Exception as e:
            error_examples += 1
            print(e, img_file)
        # if i == 1000:
        # break
    

    # 分割数据集
    num_val = 5000
    num_test = 10000
    num_test, num_val, num_train, num_trainval = split_datasets_trainval_test(all_selected_imgs, num_val, num_test, root_path)

    print("="*60)
    print(f"ccpd total: {len(all_imgs_path)}")
    print(f"ccpd num_error: {error_examples}")
    print(f"ccpd saved_num: {len(all_selected_imgs)}")
    print(f"ccpd num_error + ccpd saved_num = {error_examples + len(all_selected_imgs)}")

    print(f"num_test={num_test}, num_val={num_val}, num_train={num_train}, num_trainval={num_trainval}")
    print(f"num_test + num_trainval = {num_test + num_trainval}")
























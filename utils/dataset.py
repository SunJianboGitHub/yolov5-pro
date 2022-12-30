#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/10/08 14:00:03
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   yolov5 load datasets
'''


import os
import random
import cv2
import sys
import hashlib
import torch
import contextlib
import platform
import albumentations as alb                # version:  albumentations==1.2.0

import numpy as np
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import to_tensor, normalize

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ExifTags
# from prefetch_generator import BackgroundGenerator


from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import distributed

sys.path.append("..")

from utils.plots import draw_norm_bboxes, convert_norm_bboxs_to_pixel, draw_pixel_bboxes, convert_pixel_bboxs_to_norm



'''
    1.镶嵌增强不同于yolov5,内部提供了裁剪到img_size,还提供了直接缩放到img_size
    2.可以镶嵌4/9/16/25等, 内部还添加了alb像素增广,水平反转
    3.混合增强模式是指: 镶嵌增强与中心对齐(像素增强+水平反转)交替进行
    4.单独增强是指镶嵌增强, 镶嵌增强可拼接4个图像, 也可以传入列表,进行多种图像的拼接
    5.不增强的条件下, 只进行中心对齐(长边等比缩放到img_size)
    6.镶嵌大图到目标输出小图,有两种方式: 直接缩放和裁剪目标尺寸
'''


HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'       # 图像的后缀名
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format

MEAN = [0.485, 0.456, 0.406]                                            # ImageNet上的图像均值, RGB通道
STD  = [0.229, 0.224, 0.225]                                            # ImageNet上的图像标准差, RGB通道

COCO_MEAN = [0.471, 0.448, 0.408]                                            # COCO上的图像均值, RGB通道
COCO_STD  = [0.234, 0.239, 0.242]                                            # COCO上的图像标准差, RGB通道

# 多图像进行归一化和标准化, 转换成tensor
def transform_image_totensor(image):
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=COCO_MEAN, std=COCO_STD)
    ])
    return trans(image)



# 数据散列算法第五代, 哈希算法
def get_md5(data):  
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(pil_img):
    # 返回 exif-corrected 的PIL size
    width, height = pil_img.size  # (width, height)
    try:
        rotation = dict(pil_img._getexif().items())[orientation]
        if rotation == 6 or rotation == 8:  # rotation 270  or  rotation 90
            # exchange
            width, height = height, width
    except Exception as e:
        pass
    return width, height


class AlbPixelAugment(object):
    def __init__(self, name="alb_pixel_augmention"):
        pass

    @staticmethod
    def build():
        prob = 0.5
        trans_sequence = alb.Compose([
            # 针对图像做颜色变换以及亮度和对比度变换
            alb.OneOf([                                                                        # 这里主要是针对颜色空间的增强
                alb.HueSaturationValue(),                                                      # HSV颜色变换
                alb.RandomToneCurve(),                                                         # 通过色调曲线,改变亮区与暗区的关系
                alb.RandomBrightnessContrast(),                                                # 随机改变亮度和对比度
            ], p=prob),
            # 各种滤波器对输入图像处理
            alb.OneOf([                                                                        # 这里主要是产生各种模糊滤波
                alb.Blur(),                                                                    # 滤波
                alb.GlassBlur(),                                                               # glass滤波
                alb.GaussianBlur(),                                                            # 高斯滤波
                alb.MedianBlur(),                                                              # 中值滤波
                alb.MotionBlur(),                                                              # 运动模糊
                alb.AdvancedBlur(),                                                            # 高级滤波
                alb.ToSepia(),                                                                 # 应用sepia滤波器处理输入图像
                alb.RingingOvershoot(),                                                        # 2D正弦滤波器处理输入图像,创建过冲伪影
            ], p=prob),
            # 在图像是哪个添加各种噪音
            alb.OneOf([                                                                         # 这里主要是产生各种噪音
                # alb.Cutout(num_holes=32, max_h_size=24, max_w_size=24, p=0.5),
                alb.GaussNoise(),
                alb.ISONoise(),                                                                 # 噪音
                alb.RandomRain(),                                                               # 雨滴
                alb.RandomFog(),                                                                # 烟雾
                alb.RandomSnow(),                                                               # 雪花
                alb.MultiplicativeNoise(),                                                      # 图像直接乘以一个数值(随机噪声)
                alb.RandomSunFlare(),                                                           # 太阳耀斑
                alb.RandomGamma(),                                                              # 
                # alb.RandomGridShuffle(),                                                        # 将grid_cell进行shuffle
                alb.RandomShadow(),                                                             # 模拟图像阴影
                # alb.Spatter(),                                         # 飞溅变换,以雨泥堵塞透镜腐蚀
                # alb.augmentations.geometric.transforms.Perspective(p=0.5)
            ], p=prob),
            # 对图像每个颜色通道的相关操作
            alb.OneOf([                                                                        
                alb.ChannelDropout(),                                                           # 随机丢失通道
                alb.ChannelShuffle(),                                                           # 随机通道打乱
                alb.Posterize(),                                                                # 减少每个颜色通道的位数, 本身是8bit
                alb.RGBShift(),                                                                 # 针对每个颜色通道, 随机移动一个数值
                alb.ToGray(),
            ], p=prob),
            # 图像的整体操作,包括锐化、反转、超像素、浮雕、均衡化
            alb.OneOf([                                                                         # 这里主要做锐化相关操作
                alb.Emboss(),                                                                   # 浮雕与原图叠加
                alb.Equalize(),                                                                 # 直方图均衡化
                # alb.InvertImg(),                                                                # 反转图像的像素值, 也就是用255减去像素值
                alb.Sharpen(),                                                                  # 锐化图像之后叠加操作
                alb.UnsharpMask(),
                # alb.Solarize(),                                                                 # 反转高于阈值的所有像素值
                # alb.Superpixels(),                                                              # 超像素表示
                alb.ImageCompression(quality_lower=80, quality_upper=100),                        # 图像质量压缩
            ], p=prob),

        ])
        return trans_sequence


    @staticmethod
    def apply(cv_image):
        trans_sequence = AlbPixelAugment.build()
        image = trans_sequence(image=cv_image)["image"]
        return image



class LoadImagesAndLabels(Dataset):
    # yolov5 训练/验证数据集加载器, 用于训练与验证阶段的数据加载和增强
    cache_version = 0.6                   # 数据集标签缓存的版本 *.cache version
    def __init__(self, 
                    datas_path,           # 数据集文件的路径, 例如 /workspace/CourseCode/datasets/classic_datasets/PASCAL_VOC2007/VOC2007_trainval/trainval.txt
                    img_size=640,         # 输入图像的尺寸
                    # batch_size=16,        # 批量尺寸
                    augment=False,        # 是否进行图像增强, 图像增强分为两种，一种是mosaic，另一种是普通的增强模式(仿射变换)
                    hyp=None,             # 训练超参数
                    mixed_aug=False,       # 都是正方形训练, 所谓的混合增强其实就是等比缩放后中心对齐、mosaic两种, 但是mosaic又可以通过mosaic_num设置不同的镶嵌数目持续增强
                    cache_images=False,   # 是否缓存图像
                    mosaic_nums=4,        # 镶嵌增强时, 图像的数目, 可选4/9/16 , 可以是一个列表[4, 9, 16], 内部自动随机选择
                    border_fill_value=114,      # 边界填充值
                    prefix=""):
        self.datas_path = datas_path
        self.img_size = img_size
        # self.batch_size = batch_size
        self.augment = augment
        self.hyp = hyp
        self.mixed_aug = mixed_aug
        self.mosaic_nums = mosaic_nums                              # 镶嵌增广需要的图像数目
        self.mosaic_border = [0, 0]                                 # 计算镶嵌中心的偏移量
        self.prefix = prefix                                        # 前缀, 用于标记
        self.cache_images = cache_images                            # 当图像较少时, 缓存图像
        self.alb_pixel_aug = True if augment else None              # 使用Alb增广, 这里主要是像素增广/化学增广
        self.border_fill_value = border_fill_value                  # 边界填充值, 主要用在mosaic中的大图np.full,提高效率
        self.border_fill_tuple = border_fill_value, border_fill_value, border_fill_value        # 边界填充元组,仿射变换填充,效率高

        self.all_images_labels_info = []                            # 所有的图像对应的标签信息
        
        try:
            self.image_files = []                                                                                   # 存储所有图像的绝对路径
            self.img_dir = ""                                                                                       # 获取图像所在文件目录, 例如images 或者 JPEGImages
            cur_path = Path(self.datas_path)                                                                        # 获取trainval.txt的所在目录，将缓存标签文件和它放在同级目录
            if cur_path.is_file():                                                                                  # 传入的必须是一个文件trainval.txt
                with open(cur_path, "r") as f:                                                                      # 打开文件
                    lines = [line.strip() for line in f.readlines()]                                                # 读取所有行文件
                    parent = str(cur_path.parent) + os.sep                                                          # 这里表明trainval.txt必须和images以及labels在同级目录
                    self.img_dir = str(lines[0].split("/")[0])                                                      # 不论这里的图像文件夹是images还是JPEGImages, 都可以智能替换为labels
                    self.image_files += [parent + name for name in lines]                                           # 拼接出每一个图像的绝对路径, 并存储
            else:
                raise Exception(f"{prefix}{cur_path} is not file.")
            self.image_files = sorted(x for x in self.image_files if x.split(".")[-1].lower() in IMG_FORMATS)       # 通过后缀名，筛选出所有图像
            assert self.image_files, f'{prefix} No images found'
            self.label_files = [img_file.replace(self.img_dir, "labels").replace(img_file.split(".")[-1], "txt") for img_file in self.image_files]      # 通过替换操作, 确定每一个图像对应标签的绝对路径
        except Exception as e:
            print(f'{prefix} Error loading {self.datas_path} from {cur_path}: {e}\n{HELP_URL}')
            raise RuntimeError()

        self.build_and_cache_labels()                                                   # 读取或者简历标签缓存, 保存到本地和self.all_images_labels_info


        # 缓存图像到 RAM, 为了加速训练, 但是当数据集较大时, 不适合这种方式, 程序容易崩溃
        self.cache_images_dict = {}
        self.cache_images_file = str(Path(self.datas_path).parent) + os.sep + "caches" + os.sep + f"{self.prefix}_images_trainval.cache.npy"
        if self.cache_images:
            if os.path.exists(self.cache_images_file):
                self.cache_images_dict = dict(np.load(self.cache_images_file, allow_pickle=True)[0])
                GB = float(self.cache_images_dict["GB"])
                # print(f"{prefix} Caching images exist. use RAM={GB / 1E9:.1f}GB.")
            else:
                GB = 0                      # 需要缓存的图像的大小 单位是 bytes
                pbar = tqdm(enumerate(self.image_files), total=len(self.image_files), bar_format=BAR_FORMAT)
                for i, img_name in pbar:
                    cv_image = cv2.imread(img_name)
                    self.cache_images_dict[img_name] = cv_image
                    GB += cv_image.nbytes
                    pbar.desc = f'{prefix} Caching images ({GB / 1E9:.1f}GB.)'

                self.cache_images_dict["GB"] = GB                                                                       # 数据集保存到内存RAM所占用的大小
                np.save(self.cache_images_file, [self.cache_images_dict])                                                 # 将读取好的图像直接保存到本地, 方便下次加载
            

    def __len__(self):
        return len(self.all_images_labels_info)

    def __getitem__(self, index):
        if self.augment:                                   # 在训练阶段, 需要进行增强操作
            if self.mixed_aug:                             # 混合增强, 包括镶嵌增强以及中心对齐(像素增强)
                if random.random() < 0.5:                  # 此处, 进行马赛克增强
                    if isinstance(self.mosaic_nums, int):
                        image, norm_bboxes = self.load_mosaic(index, self.mosaic_nums)
                    elif isinstance(self.mosaic_nums, list):
                        image, norm_bboxes = self.load_mosaic(index, self.mosaic_nums[random.randint(0, len(self.mosaic_nums)-1)])
                    else:
                        image, norm_bboxes = self.load_mosaic(index, num=4)
                else:                                                               # 此处进行中心对齐、alb增广
                    image, norm_bboxes = self.load_center_image(index)              # 先长边等比缩放, 在进行边缘填充到长边尺度

                    if random.random() < 0.5:                                                   # 左右翻转
                        image, norm_bboxes = self.horizontal_flip(image, norm_bboxes)
                    if self.alb_pixel_aug:
                        image = AlbPixelAugment.apply(image)                            # alb 像素增广
            else:                                                                   # 只进行马赛克增强, 这里的缺点是anchor和增强数据不匹配, 在anchor-free场景中最合适
                if isinstance(self.mosaic_nums, int):
                    image, norm_bboxes = self.load_mosaic(index, self.mosaic_nums)
                elif isinstance(self.mosaic_nums, list):
                    image, norm_bboxes = self.load_mosaic(index, self.mosaic_nums[random.randint(0, len(self.mosaic_nums)-1)])
                else:
                    image, norm_bboxes = self.load_mosaic(index, num=4) 
        else:                                                                       # 在验证测试阶段, 不需要增强, 直接等比缩放+填充边界到长边大小
            image, norm_bboxes = self.load_center_image(index)

        return image, norm_bboxes



    def load_image(self, img_idx):
        """
            ### 加载图像说明
                - 等比例缩放, 长边缩放到 img_size=640
                - yolov5中, 是训练阶段使用
            ### 返回值说明
                - cv格式的图像
                - 归一化的box标签数据
                - 缩放后的图像的宽度和高度
        """
        img_file, norm_labels_info, (img_width, img_height) = self.all_images_labels_info[img_idx]              # 根据索引获取需要加载的图像信息
        # image = cv2.imread(img_file)                                                                            # 读取当前图像
        if img_file in self.cache_images_dict.keys():                                                           # 如果图像被存储在RAM内存中,直接读取
            image = self.cache_images_dict[img_file]
        else:
            image = cv2.imread(img_file) 

        scale = min(self.img_size / img_width, self.img_size / img_height)                                      # 找到最小的缩放尺度
        if scale != 1:
            if not self.augment and scale < 1:                                                                  # 如果不需要增广(评估测试阶段), 并且缩放系数小于1, 就使用效果较好的差值方式
                interp_mode = cv2.INTER_AREA                                                                    # 区域插值, 效果好, 速度慢
            else:
                interp_mode = cv2.INTER_LINEAR                                                                  # 线性插值, 速度快, 效果也还可以

            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=interp_mode)                    # 图像等比例缩放
        img_resized_height, img_resized_width = image.shape[:2]                                                 # 获取缩放之后图像的宽高
        return image, norm_labels_info.copy(), (img_resized_width, img_resized_height)


    def load_center_image(self, img_idx):
        """
            ### 函数说明
                - 1. 先进行等比例缩放, 长边缩放到 img_size=640
                - 2. 再将缩放后的图像中心, 平移到目标图像的中心
                - 3. yolov5的验证评估阶段采用的这种方式
            ### 实现方式
                - 1. 边界填充
                - 2. 仿射变换
        """
        image, norm_labels_info, (img_width, img_height) = self.load_image(img_idx)                             # 读取图像, 得到等比缩放后的图像以及标签
        # draw_norm_bboxes(image, norm_labels_info[:, 1:], color=(0, 0, 255), thickness=1)                      # 用于验证转换之后的边界框是否完全重合

        offset_x, offset_y = (self.img_size/2 - img_width/2), (self.img_size/2 - img_height/2)                  # 仿射变换的偏移量
        M = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y]
        ], dtype=np.float32)
        image = cv2.warpAffine(image, M, (self.img_size, self.img_size), borderMode=cv2.BORDER_CONSTANT, borderValue=self.border_fill_tuple)
        # real_cx = (cx * w + offset_x), norm_cx = real_cx / img_size
        norm_labels_info[:, 1: 3] = (norm_labels_info[:, 1: 3] * np.array([img_width, img_height]) + np.array([offset_x, offset_y])) / np.array([self.img_size, self.img_size])               # 针对中心坐标cx,cy的计算
        norm_labels_info[:, 3:  ] = (norm_labels_info[:, 3:  ] * np.array([img_width, img_height])) / np.array([self.img_size, self.img_size])                                                # 针对边界框w, h的计算

        return image, norm_labels_info


    def load_mosaic(self, img_idx, num=4):
        """
            ### 函数说明
                - 马赛克增广, yolov5最先提出, yolov4论文先使用, yolov5团队认为yolov4是依靠mosaic做到SOTA, 因此马上推出自己的yolov5,  两者各有千秋
                - 先确定1个img_idx为指定的图, 再数据集中随机选择n个随机图, 拼接为马赛克
                - yolov5中一共出现了两种马赛克增强, mosaic4 和 mosaic9, 每种都有自己的排布方式, 都是先排中心
                - 这里将会采用仿射变换实现马赛克增广
                - 这里采用的方式不完全等同于yolov5
            ### 返回值
                - image[img_size, img_size]
                - norm_labels_info[Nx5]
        """
        num_images = len(self.all_images_labels_info)                                       # 获取数据集的总数目
        indices = [img_idx] + random.choices(range(num_images), k=num-1)                    # 获取马赛克增广的所有图像的索引
        random.shuffle(indices)

        # 确定每一个图像的偏移系数
        n = int(np.sqrt(num))                                                                                                   # 每一行每一列的图像数目
        assert num == n**2, f"num={num}, parameter error."

        large_image = np.full((self.img_size * n, self.img_size * n, 3), self.border_fill_value, dtype=np.uint8)                # 填充值,效率更高,填充元组并不好
        merge_mosaic_pixel_bboxes = []

        record_real_offset = np.zeros((n+1, n+1, 2))                                                                            # 存储的是图像的右下角坐标 right  bottom
        # 记录每张图像的左上角坐标点的实际偏移量
        for row in range(n):                                                                                                    # 遍历每一个马赛克增广的每一块
            last_bottom_max_limit = max(record_real_offset[row, :, 1])
            for col in range(n):
                idx = row * n + col                                                                                             # 当前块应该填充的图像的索引
                image, normal_labels_info, (img_width, img_height) = self.load_image(indices[idx])                              # 加载等比缩放后的图像
                
                if self.alb_pixel_aug:
                    image = AlbPixelAugment.apply(image)                                                                            # alb增广,只对图像增广
                if random.random() < 0.5:
                    image, normal_labels_info = self.horizontal_flip(image, normal_labels_info)                                 # 水平翻转

                # draw_norm_bboxes(image, normal_labels_info[:, 1:], color=(0, 0, 255), thickness=6)

                pre_right_max_limit = record_real_offset[row+1, col, 0]
                offset_x = random.randint(0, self.img_size * (col + 1) - pre_right_max_limit - img_width)         # 计算当前图像的x偏移量
                offset_y = random.randint(0, self.img_size * (row + 1) - last_bottom_max_limit - img_height)                  # 计算当前图像的y偏移量
   

                M = np.array([
                        [1, 0, pre_right_max_limit + offset_x],
                        [0, 1, last_bottom_max_limit + offset_y]
                    ], dtype=np.float32)
                cv2.warpAffine(image, M, (self.img_size * n, self.img_size * n), dst=large_image, borderMode=cv2.BORDER_TRANSPARENT)

                pixel_bboxes = convert_norm_bboxs_to_pixel(normal_labels_info, img_width, img_height)                          # 将归一化边界框转换为像素单位,[class_id, x, y, r, b]
                pixel_bboxes = pixel_bboxes + [0, M[0, 2], M[1, 2], M[0, 2], M[1, 2]]                                          # 平移边界框到指定位置
                merge_mosaic_pixel_bboxes.append(pixel_bboxes)                                                                 # 合并所有的框, 维度是 [9, M, 5] 

                record_real_offset[row+1, col+1] = [pre_right_max_limit + img_width + offset_x, last_bottom_max_limit + img_height + offset_y]

                # break
            # break
        merge_mosaic_pixel_bboxes = np.concatenate(merge_mosaic_pixel_bboxes, axis=0)                                          # 将所有的bbox合并, 维度是[Nx5]
        # np.clip(merge_mosaic_pixel_bboxes[:, 1:], a_min=0, a_max=self.img_size*n-1, out=merge_mosaic_pixel_bboxes[:, 1:])      # 限制边界框到指定范围内
                
        # 到此为止, mosaic已经完成, 此时的图像大小是 img_size * n
        # 接下来, 我们希望输出的图像尺度是 img_size, 这里有两种方案
        # 方案1: 直接 resize 到 img_size, 好处是正样本多, 目标尺度变小,利于小目标检测, 缺点是可能有些边界框缩放之后太小会被忽略
        # 方案2: 将大图缩放到指定大小, 裁剪出 img_size大小的图像作为输出, 好处是对边界框的改变较小, 更适合原始的 anchor
        # 在这里可以根据不同概率值, 选择其中一种策略
        if random.random() < 1:
            scale = 1. / n                                                                                                          # 直接缩放到目标尺寸的系数
            M = np.array([
                        [scale, 0, 0],
                        [0, scale, 0]
                    ], dtype=np.float32)                                                                                            # 仿射变换矩阵
        else:
            scale = random.uniform(1./n, 1+(1./n))                                                                                  # 当 n=2时，就是0.5-1.5
            offset = self.img_size * 0.5 - self.img_size * n * scale * 0.5                                                          # 中心对齐的偏移量
            M = np.array([
                        [scale, 0, offset],
                        [0, scale, offset]
                    ], dtype=np.float32)

        merge_mosaic_image = cv2.warpAffine(large_image, M, (self.img_size, self.img_size), 
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.border_fill_tuple)    # 对图像进行仿射变换

        # 开始转换边界框信息, 因为缩放平移之后，有些边界框可能超出范围了，需要裁减掉
        num_targets = len(merge_mosaic_pixel_bboxes)                                # 计算当前合并大图所包含的目标数
        merge_mosaic_norm_bboxes = np.zeros((0, 5))                                 # 如果没有目标返回一个带维度的0
        if num_targets > 0:                                                         # 代表单签合并的大图上存在目标
            # 内存排布知识
            # N x 4 -> left, top, right, bottom, left, top, right, bottom, left, top, right, bottom
            #       -> reshape(N*2, 2)
            #       -> left, top
            #       -> right, botom
            #       -> left, top
            #       -> right, botom
            # 把所有的边界框变成了一行一个点, 两行是1个边界框
            tmp_targets = np.ones((num_targets*2, 3))                                           # 用于将标签数据与仿射变换矩阵相乘, 维度是 2Nx3
            tmp_targets[:, :2] = merge_mosaic_pixel_bboxes[:, 1:].reshape(num_targets*2, 2)     # 维度是 2Nx3
            # tmp_targets 维度是 2Nx3
            # M           维度是 2x3
            # outout      维度是 2Nx2
            affine_pixel_bboxes = merge_mosaic_pixel_bboxes.copy()                              # 复制一份标签，其实是为了复制class_id
            affine_pixel_bboxes[:, 1:] = (tmp_targets @ M.T).reshape(num_targets, 4)            # 得到仿射变换后的边界框
            # 处理边界框
            # 1. 超出范围的
            # 2. 无效的, 剩余部分面积小于指定比例的或者像素小于多少的
            np.clip(affine_pixel_bboxes[:, 1:], a_min=0, a_max=self.img_size-1, out=affine_pixel_bboxes[:, 1:])         # 裁剪超出范围的边界框
            affine_bbox_w = affine_pixel_bboxes[:, 3] - affine_pixel_bboxes[:, 1] + 1                                   # 计算裁剪之后的边界框的宽
            affine_bbox_h = affine_pixel_bboxes[:, 4] - affine_pixel_bboxes[:, 2] + 1                                   # 计算裁剪之后的边界框的高
            origin_bbox_w = merge_mosaic_pixel_bboxes[:, 3] - merge_mosaic_pixel_bboxes[:, 1] + 1                       # 计算缩放裁剪之前的边界框的宽
            origin_bbox_h = merge_mosaic_pixel_bboxes[:, 4] - merge_mosaic_pixel_bboxes[:, 2] + 1                       # 计算缩放裁剪之前的边界框的高
            affine_bbox_area = affine_bbox_w * affine_bbox_h                                                            # 计算裁剪之后的边界框的面积
            origin_bbox_area = origin_bbox_w * origin_bbox_h                                                            # 计算原始的边界框的面积
            wh_ratio = np.maximum(affine_bbox_w/(affine_bbox_h+1e-6), affine_bbox_h/(affine_bbox_w+1e-6))               # 高宽比、宽高比的最大值
            # 边界框的保存条件分析
            # 1. 裁剪后的框的宽度和高度都必须超过2个像素, 但是感觉这里与聚类anchor时(其中一个大于2)的不太一样？
            # 2. 裁剪后的面积 / 裁剪前的面积 > 0.2, 因为会存在只裁剪到大目标的局部特征, 很难判断是某一个目标，这个值很难确定, 也很难解决根本
            # 3. max(宽高比, 高宽比)  < 20, 因为会存在裁剪成一条细条线的情况, 是为了去除这种影响, 但是大目标这种也不太合适, 也很难解决根本
            # 所以我个人觉得直接缩放到目标尺寸是最好的, 不会存在裁剪歧义的问题
            keep_indices = (affine_bbox_w > 2) & (affine_bbox_h > 2) & \
                           (affine_bbox_area/(origin_bbox_area * scale + 1e-6) > 0.2) & \
                           (wh_ratio < 20)
            affine_pixel_bboxes = affine_pixel_bboxes[keep_indices]                                 # 裁剪过滤之后的边界框, 格式是 [class_id, lsft, top, right, bottom]
            merge_mosaic_norm_bboxes = convert_pixel_bboxs_to_norm(affine_pixel_bboxes, self.img_size, self.img_size)       # pixel_box归一化为norm_box


            # print(f"origin_num_targets={num_targets}, clip_num_targets={len(affine_pixel_bboxes)}")

            pass

        # return large_image, merge_mosaic_pixel_bboxes
        return merge_mosaic_image, merge_mosaic_norm_bboxes


    def horizontal_flip(self, image, norm_bboxes):
        """
            ### 函数说明
                - 对图像和边界框进行水平翻转
            ### 参数
                - image[ndarray]: 输入图像
                - norm_bboxes[Nx5]: 提供的是归一化后的边界框信息,格式是 [class_id, cx, cy, w, h]
            ### 返回值
                - image
                - norm_bbox
        """
        # flipCode = 1 ，   水平，也就是x轴翻转
        # flipCode = 0，    垂直，也就是y轴翻转
        # flipCode = -1，   对角翻转，x和y都发生翻转
        image = cv2.flip(image, flipCode=1)                                             # 图像的水平翻转
        flip_norm_bboxes = norm_bboxes.copy()                                           # 拷贝标签数据
        img_w = image.shape[1]                                                          # 计算图像的宽度
        flip_norm_bboxes[:, 1] = (img_w - 1) / img_w - flip_norm_bboxes[:, 1]           # 注意这里并不是 1-, 去掉边界
        return image, flip_norm_bboxes


    def hsv_augment(self, image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
        """
            ### 函数说明
                - 对图像进行HSV颜色空间增广
            ### 参数
                - image[ndarray]: 输入图像
                - h_gain[float]:  色调增益, 最终的增益系数为   random(-1, +1) * h_gain + 1
                - s_gain[float]:  饱和度增益, 最终的增益系数为 random(-1, +1) * s_gain + 1
                - v_gain[float]:  亮度增益, 最终的增益系数为   random(-1, +1) * v_gain + 1
            ### 返回值
                - image[ndarray]
        """
        h_gain = np.random.uniform(-1, +1) * h_gain + 1                     # 最终色调的增益系数
        s_gain = np.random.uniform(-1, +1) * s_gain + 1                     # 最终饱和度的增益系数
        v_gain = np.random.uniform(-1, +1) * v_gain + 1                     # 最终亮度的增益系数

        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))   # 图像转换为HSV, 并分解为 H/S/V 3个通道
        # hue        ->  值域 0 - 179
        # saturation ->  值域 0 - 255
        # value      ->  值域 0 - 255
        dtype = image.dtype
        lut_base = np.arange(0, 256)
        lut_hue = ((lut_base * h_gain) % 180).astype(dtype)
        lut_sat = np.clip(lut_base * s_gain, 0, 255).astype(dtype)
        lut_val = np.clip(lut_base * v_gain, 0, 255).astype(dtype)

        # cv2.LUT(index, lut)
        changed_hue = cv2.LUT(hue, lut_hue)
        changed_sat = cv2.LUT(sat, lut_sat)
        changed_val = cv2.LUT(val, lut_val)
        img_hsv = cv2.merge((changed_hue, changed_sat, changed_val))

        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


    def build_and_cache_labels(self):
        # 检查标签缓存
        cache_name = get_md5(self.datas_path)
        cache_path = str(Path(self.datas_path).parent) + os.sep + "caches" + os.sep + f"{self.prefix}_labels_{cache_name}.cache.npy"
        try:
            self.all_images_labels_info = np.load(cache_path, allow_pickle=True)
            # print(f"Load labels from cache: {cache_path}")
        except Exception as e:
            print(f"Build labels and save to cache: {cache_path}")
            if not os.path.exists(Path(cache_path).parent):
                os.makedirs(Path(cache_path).parent)
            self.check_images_and_cache_labels(cache_path)                                  # 执行缓存标签的操作, 读取的标签信息存在self.all_images_labels_info
        pass


    def check_images_and_cache_labels(self, cache_path):
        """
            ### 数据检查:
                - 1.图像是否损坏, 如果损坏, 直接抛出异常
                - 2.检查图像大小是否过小, 如果太小, 直接抛出异常
                - 3.加载标准信息, 并保存起来
        """
        desc = f"{self.prefix}Scanning '{Path(self.datas_path).parent / Path(self.datas_path).stem}' images and labels..."
        pbar = tqdm(zip(self.image_files, self.label_files), total=len(self.image_files),
                        bar_format=BAR_FORMAT)
        num_error = 0
        for img_file, label_file in pbar:                                           # 遍历整个数据集, 检查图像和标签数据是否存在异常,并缓存标签
            try:
                pil_image = Image.open(img_file)                                    # PIL 读取图像
                pil_image.verify()                                                  # 检查图像是否存在异常, 如果存在异常会直接抛出
                image_width, image_height = exif_size(pil_image)                    # 矫正图像, 获取图像的高度和宽度
                assert image_height > 9 and image_width > 9, f"Image size is too small{image_width} x {image_height}"       # 去掉太小的图像

                with open(label_file, "r") as f:                                    # 打开标签文件
                    lines = f.readlines()                                           # 读取所有行
                    lines = [line.strip().split(" ") for line in lines]             # 每一行分割为一个列表
                    if len(lines) == 0:                                             # 如果行数为零，代表该图像没有标注信息
                        label_info = np.zeros((0, 5), dtype=np.float32)             # cls, x, y, w, h
                    else:
                        label_info = []
                        for i in range(len(lines)):
                            lines[i] = np.array([float(item) for item in lines[i]])
                            assert len(lines[i]) >= 5,  f'labels require 5 columns, {lines[i].shape} columns detected'
                            assert (lines[i] >= 0).all(), f'negative label values {lines[i][lines[i] < 0]}'
                            assert (lines[i][1:] <= 1).all(), f'non-normalized or out of bounds coordinates.'
                            label_info.append(lines[i])
                        label_info = np.array(label_info)
                    self.all_images_labels_info.append([img_file, label_info, [image_width, image_height]])
            except Exception as e:
                num_error += 1
                print(f"num_error={num_error}")
        np.save(cache_path, self.all_images_labels_info)
        return self.all_images_labels_info

    @staticmethod
    def collate_fn(batch):
        """
            ### 函数说明
                - 该函数执行是在 dataset.__getitem__之后, dataloader获取数据之前
                - 所谓的获取之前是指 for batch_idx, (images, labels) in enumerate(dataloader):
                - 这里我们需要在标签数据的0位置添加一个img_id
            ### 返回值
                - 
        """
        # batch = [[image, labels], [image, labels]]
        images, labels = zip(*batch)
        batch_images = []
        batch_labels = []
        for idx, (image, label) in enumerate(zip(images, labels)):
            new_label = np.zeros((len(label), 6))                   # 最终的输出维度是 [M x 6]
            new_label[:, 0] = idx                                   # 在第一个维度添加图像的ID
            new_label[:, 1:] = label[:, [0, 1, 2, 3, 4]]            # 后面的数据直接赋值
            batch_labels.append(new_label)                          # 所有标签添加列表

            # OpenCV格式的图像,维度是[H, W, C], 而且C通道的排布是 BGR, 但是我们通常用RGB来训练,因此应转换一下
            # 这里还需要标准化, 采用torchvision.transform.functional.normalize
            # torchvision.transform.functional.to_tensor(), BGR->RGB,并进行归一化
            # norm_image = normalize(to_tensor(image[..., ::-1]), mean=MEAN, std=STD)
            # 这里要特别注意, 需要连续化
            norm_image = np.ascontiguousarray(image.transpose(2, 0, 1)[::-1])             # 先转换为 RGB, 然后再连续化
            # norm_image = np.ascontiguousarray(image[..., ::-1])             # 先转换为 RGB, 然后再连续化
            # norm_image = transform_image_totensor(norm_image)
            
            # image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # image = np.ascontiguousarray(image)
            batch_images.append(torch.from_numpy(norm_image))    
            
        
        # 准备visual_info, 训练过程中用visdom来显示的东西, image, label, img_id
        visual_img_id = random.randint(0, len(images)-1)
        visual_image, visual_label = images[visual_img_id], labels[visual_img_id]
        
        batch_labels = np.concatenate(batch_labels, axis=0)         # 合并所有的标签为 [N x 6]
        batch_labels = torch.FloatTensor(batch_labels)              # 转换为torch float tensor
        batch_images = torch.stack(batch_images, dim=0)
        
        return batch_images, batch_labels, (visual_img_id, visual_image, visual_label)


# 创建数据集加载器
def create_dataloader(datas_path, hyp=None, shuffle=True, augment=True, mixed_aug=False, cache_images=False, mosaic_nums=[4, 9], prefix="", batch_size=32, img_size=640, num_workers=16, border_fill_value=114):
    data_sets = LoadImagesAndLabels(datas_path=datas_path, 
                                         img_size=img_size, 
                                         augment=augment,
                                         hyp=hyp,
                                         mixed_aug=mixed_aug,
                                         cache_images=cache_images,
                                         mosaic_nums=mosaic_nums,
                                         prefix=prefix,
                                         border_fill_value=border_fill_value)
    data_loader = DataLoader(data_sets, batch_size=batch_size, shuffle=shuffle, 
                                 num_workers=num_workers, collate_fn=data_sets.collate_fn, pin_memory=True)
    
    return data_loader, data_sets











if __name__ == "__main__":
    sys.path.append("..")
    from matplotlib import pyplot as plt
    from models.yolo import YoloV5
    
    
    
    # model = YoloV5(1, "/workspace/yolov5-pro/models/yolov5s.yaml")
    # model.cuda()
    # model.train()

    
    
    root = "/workspace/datasets/PASCAL_VOC2007/VOC2007_trainval/debug.txt"
    train_datasets  = LoadImagesAndLabels(root, prefix="debug",augment=False, cache_images=False)
    train_dataloader = DataLoader(train_datasets, 16, shuffle=False, num_workers=15, collate_fn=train_datasets.collate_fn, pin_memory=True, prefetch_factor=2)
    # image, norm_label, (w, h) = my_dataset.load_image(1)
    # draw_norm_bboxes(image, norm_label[:, 1:], color=(0, 0, 255), thickness=3)
    # cv2.imwrite("test_show.jpg", image)


    iter_id = 0
    pbar = tqdm(train_dataloader, desc="Reading batch images and labels for train... ")
    for batch_images, batch_labels, visual_info in pbar:
        # batch_images, batch_labels = batch_images.cuda(non_blocking=True), batch_labels.cuda(non_blocking=True)

        # print(batch_labels)
        id = 1
        image = batch_images[id].numpy()
        labels = batch_labels.numpy()
        
        image = np.ascontiguousarray(image.transpose(1, 2, 0)[..., ::-1])
        print(image.shape)
        cv2.imwrite("ssssssssssssssss.jpg", image)
        # print(image)
        for i in range(len(labels)):
            if int(labels[i][0]) == id:
                x, y, w, h = labels[i][2:] * 640
                left = int((x - (w-1) /2) * 1)
                top = int((y - (h-1)/2) * 1)
                right = int((x + (w-1) /2) * 1)
                bottom = int((y + (h-1)/2) * 1)
                cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.imwrite("hhhhhhhhhhhhhhhh.jpg", image)

        break
   




































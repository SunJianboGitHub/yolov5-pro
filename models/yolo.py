#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   yolo.py
@Time    :   2022/09/27 14:41:45
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   yolov5 backbone neck head modules
'''

import os
import sys
import yaml
import contextlib
import platform
import torch
import torch.nn as nn
from pathlib import Path


sys.path.append("..")


from models.common import *
from utils.general import make_divisible
from utils.general import _single_instance_logger as LOGGER
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights



class Detect(nn.Module):
    
    # 是否导出为onnx
    export = False
    
    def __init__(self, num_cls=80, anchors=(), reference_channels=(), inplace=True):
        super().__init__()
        self.num_cls = num_cls                                                          # 检测的类别数
        self.strides = torch.FloatTensor([8, 16, 32])                                   # 每一个预测层的步长(缩放的倍数)
        self.anchors = torch.FloatTensor(anchors).view(3, 3, 2) / self.strides.view(3, 1, 1)
        self.num_anchor_per_level = len(anchors[0]) // 2                                # 每一个检测头包含的anchor的数目
        self.num_layer = len(anchors)                                                   # 包含多少个检测头
        
        # 当只有一个检测类别时,直接用objectness代替类别概率计算loss
        # 但是网络层的输出通道数任然是 5+num_cls, 也就是6个通道
        self.num_output_per_level = (5 + self.num_cls) * self.num_anchor_per_level      # 每一个检测头输出的通道数

        self.detect_heads = nn.ModuleList([nn.Conv2d(c_in, self.num_output_per_level, 1, 1) for c_in in reference_channels])    # 多个检测头输出
        self.inplace = inplace

        # print(f"\nDetect layer num_cls == {self.num_cls}\n")
        
        # 初始化检测头的偏置参数
        self._initialize_biases()
        
        
    # 初始化检测头的偏置参数，具体原理参照Focal Loss论文，保证输出概率接近0.01
    def _initialize_biases(self, cf=None):
        for head, s in zip(self.detect_heads, self.strides.numpy()):        # 遍历每一个检测层
            bias = head.bias.view(self.num_anchor_per_level, -1)            # conv.bias(75) -> (3, 25)
            bias.data[:, 4] += math.log(8 / (640 / s) ** 2)                 # obj (8 objects per 640 image)
            bias.data[:, 5:5 + self.num_cls] += math.log(0.6 / (self.num_cls - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            head.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
        


    def forward(self, x_list):
        """
            x_list: x_list 是一个list , 它是传递到检测头的各层输出, 包括 P3 P4 P5
        """
        self.set_device = x_list[0].device                                  # 推理阶段使用
        self.anchors = self.anchors.to(self.set_device)                                   # 将anchor转移到cuda上
        self.strides = self.strides.to(self.set_device)
        self.dtype = self.anchors.dtype
        
        if self.training or self.export:                                        # 训练阶段，或者onnx导出时
            for ilevel, detect_head in enumerate(self.detect_heads):
                x_list[ilevel] = detect_head(x_list[ilevel])                    # 卷积层, 一个检测头的输出, 维度是 [bs, 255, 80, 80] 
            return x_list
        else:                                                                   # 这里仅限pytorch直接推理
            
            inference_output = []                                               # 推理阶段转换输出
            
            
            for ilevel, detect_head in enumerate(self.detect_heads):
                x_list[ilevel] = detect_head(x_list[ilevel])                # 卷积层, 一个检测头的输出, 维度是 [bs, 255, 80, 80] 
                batch_size, all_chs, fmap_h, fmap_w = x_list[ilevel].shape        # 获取当前预测层的维度信息, [bs, 255, 80, 80]
                x_list[ilevel] = x_list[ilevel].view(batch_size, self.num_anchor_per_level, int(all_chs / self.num_anchor_per_level), fmap_h, fmap_w)  # 维度是[bs, 3, 85, 80, 80]
                x_list[ilevel] = x_list[ilevel].permute(0, 1, 3, 4, 2).contiguous()

                grid_xy, anchor_wh_grid = self._make_grid(fmap_h, fmap_w, ilevel)                    # 制作网格, 恢复实际中心与宽高
                xy, wh, conf = x_list[ilevel].sigmoid().split((2, 2, 1 + self.num_cls), 4)      # 将所有数据在第4个维度切分 xy,wh, conf三个部分,单独恢复

                real_xy = (xy * 2.0 - 0.5 + grid_xy) * self.strides[ilevel]                     # 恢复到原图上的边界框中心点
                real_wh = torch.pow(wh * 2.0, 2.0) * anchor_wh_grid                             # 恢复到原图尺寸的边界框宽高
                y = torch.cat((real_xy, real_wh, conf), 4)                         # 维度是 [bs, num_anchor, h, w, 5+cls]

                inference_output.append(
                            y.view(batch_size, self.num_anchor_per_level * fmap_h * fmap_w, 5 + self.num_cls))

            return torch.cat(inference_output, dim=1)


    def _make_grid(self, fmap_h=80, fmap_w=80, i=0):
        shape = 1, self.num_anchor_per_level, fmap_h, fmap_w, 2         # grid_xy的形状
        y, x = torch.arange(fmap_h, device=self.set_device, dtype=self.dtype), torch.arange(fmap_w, device=self.set_device, dtype=self.dtype)
        grid_y, grid_x = torch.meshgrid(y, x)                          # 产生网格,用于恢复预测中心
        grid_xy = torch.stack((grid_x, grid_y), dim=2).expand(shape)    # 维度信息是[1, 3, 80, 80, 2], 注意这里是先宽度再高度,为了与预测的xy,wh匹配
        anchor_wh_grid = (self.anchors[i] * self.strides[i]).view(1, self.num_anchor_per_level, 1, 1, 2).expand(shape)                 # 特征图尺度下的anchor的宽高尺寸
        return grid_xy, anchor_wh_grid




class YoloV5(nn.Module):
    """
        ### 参数
            - yaml_cfg_file:    网络结构配置文件
            - new_anchors:      如果对数据集进行了聚类, 这里重置anchors
    """
    def __init__(self, yaml_cfg_file, new_anchors=None):
        super().__init__()
        with open(yaml_cfg_file, "r") as f:
            self.yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        self.num_classes = self.yaml_dict['nc']                                     # 类别数
        if new_anchors:                                                             # 如果针对数据集进行的聚类，可以在这里重新设置
            self.yaml_dict["anchors"] = new_anchors
        self.anchors = self.yaml_dict["anchors"]                                    # 这时的anchors是数据集上重新聚类得到的

        self.model, self.saved_index = parse_model(self.yaml_dict, 3)
        # print("self.saved_index = ", self.saved_index)
        # print("self.model = ", self.model)
        
        
        # 模型权重初始化, yolov5在这里只是进行默认初始化,设置一些参数而已
        # 通常需要针对检测头进行合理初始化,才可加速模型收敛
        initialize_weights(self.model)
        
        
        

    def forward(self, x, visualsize=False):
        x = self._forward_once(x, visualsize)
        return x

    
    def _forward_once(self, x, visualize=False):
        saved_layers = []                                               # 保存网络层，不需保存的设置为None
        for module_instance in self.model:
            if module_instance.from_index != -1:                        # 该网络层的输入不是来自于之前的一层
                if isinstance(module_instance.from_index, int):         # 说明该网络层只依赖一个输入
                    x = saved_layers[module_instance.from_index]        # 从保存列表中取出输入层
                else:                                                   # 说明该网络层不止一个输入，很可能是concat层
                    x_list = []
                    for i in module_instance.from_index:
                        if i == -1:
                            xvalue = x
                        else:
                            xvalue = saved_layers[i]
                        x_list.append(xvalue)
                    x = x_list
            
            x = module_instance(x)
            if module_instance.layer_index in self.saved_index:
                saved_layers.append(x)
            else:
                saved_layers.append(None)

            if visualize == True:
                feature_visualization(x, module_instance.module_type, module_instance.layer_index, max_n=32)
        return x


    # 吸BN, fuse model Conv2d + BN  -> Conv2d
    # 这里只合并了Conv卷积里面的BN，concat之后的有些BN并没有合并
    def fuse(self):
        print("Fusing Conv2d+BN to  Conv2d layers...")
        for m in self.model.modules():                                          # 遍历每一个网络层
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):              # 如果卷积是带BN的
                m.conv = fuse_conv_and_bn(m.conv, m.bn)                         # 将卷积替换为合并后的卷积参数
                delattr(m, "bn")                                                # 删除该网络层中的bn
                m.forward = m.forward_fuse                                      # 修改卷积合并后的前向推理
        print("Finished!")
        



def eval_strings(item):
    with contextlib.suppress(NameError):
        return eval(item) if isinstance(item, str) else item

def parse_args(value):
    try:
        return eval(value)
    except Exception as e:
        return value


def parse_model(yaml_dict, input_channel=3):
    """
        功能: 解析yolov5s.yaml文件, 搭建网络模型
        yaml_dict:          yolov5模型配置参数
        input_channel:     输入通道数, 通常为3
    """
    LOGGER.info(f"\n{'':>38}{'layer_index':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, num_cls = yaml_dict['anchors'], yaml_dict['nc']                                        # 设置的anchor和检测类别数目
    depth_multiple, width_multiple = yaml_dict['depth_multiple'], yaml_dict['width_multiple']       # 网络深度和宽度的扩增倍数

    num_anchor_per_level = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors         # 每一个特征层的anchor数目
    num_output_per_level = (5 + num_cls) * num_anchor_per_level                                     # 每层输出的特征图的深度

    layers_cfg_list = yaml_dict['backbone'] + yaml_dict['head']                                     # 所有的网络层配置
    layers_channel_list = [input_channel]                                                           # 用来存储所有网络层的输出通道数
    layers_list = []                                                                                # 用来存储所有的layer
    saved_layers_index = []                                                                         # 需要保存中间结果的层，用来拼接使用的

    for layer_index, (from_index, repeat, module_name, args) in enumerate(layers_cfg_list):         # 遍历每一个网络层的配置参数
        module_name = eval_strings(module_name)           # 解析 网络层模块的名称
        args = [parse_args(item) for item in args]                     # 解析 网络层的参数

        repeat = n_ =  max(round(repeat * depth_multiple), 1) if repeat > 1 else repeat                   # 该模块在深度方面的增益
        if module_name in [Conv, Bottleneck, BottleneckCSP, C3, SPP, SPPF, Focus]:                  # 如果是这些模块其中之一
            input_channel = layers_channel_list[from_index]                                         # 输入通道数为之前层的输出通道数
            output_channel = args[0]                                                                # 第一个参数是输出通道数
            if output_channel != num_output_per_level:                                              # 如果不是输出层，其实这里没必要写
                output_channel = make_divisible(output_channel * width_multiple, 8)

            args = [input_channel, output_channel, *args[1:]]                                       # args[0]是output_channel,args[1:]属于layer的特定参数
            if module_name in [BottleneckCSP, C3]:                                                  # 重复的模块
                args.insert(2, repeat)                                                              # 将重复次数加入参数列表[c_in, c_out, repeat, *args[1:]]
                repeat = 1
        elif module_name is Concat:                                                                 # 如果是拼接层
            output_channel = 0
            for index in from_index:
                if index != -1:
                    index += 1
                output_channel += layers_channel_list[index]

        elif module_name is Detect:                                                                 # 如果是检测头层
            reference_channels = [layers_channel_list[idx+1] for idx in from_index]                   # 各个检测头层的输入通道数
            args = [num_cls, anchors, reference_channels]
        
        else:                                                                                       # 这里主要是指上采样层nn.Upsample
            output_channel = layers_channel_list[from_index]

        if repeat > 1:                                                                              # 构建模块
            module_instance = nn.ModuleList([module_name(*args) for _ in range(repeat)])            # 重复模块
        else:
            module_instance = module_name(*args)                                                    # 单一模块

        num_params = sum(x.numel() for x in module_instance.parameters())                           # 计算当前网络层的参数量
        module_type = str(module_name)[8:-2].replace('__main__.', '')                               # module type

        module_instance.from_index = from_index
        module_instance.layer_index = layer_index
        module_instance.module_type = module_type
        module_instance.num_params = num_params

        layers_channel_list.append(output_channel)                                                  # 记录当前模块的输出通道数

        if not isinstance(from_index, list):
            from_index = [from_index]

        saved_layers_index.extend(filter(lambda idx: idx != -1, from_index))                        # 如果不是-1的层索引需要记录下来，前向传播时需要记录它们的输出

        LOGGER.info(f'{layer_index:>3}{str(from_index):>18}{n_:>3}{num_params:10.0f}  {module_type:<40}{str(args):<30}')  # print

        layers_list.append(module_instance)

    return nn.Sequential(*layers_list), sorted(saved_layers_index)




def export_onnx(model):
    model.model[-1].export=True                                             # 处理Detect层的输出，是未经后处理的，维度是 1*255*80*80
    # model.model[0].forward = model.model[0].forward_export                  # 处理Focus网络层，推理时不导出，直接在前处理时进行Focus层
    torch.onnx.export(
        model,                                                              # 需要导出的模型
        (torch.zeros((1, 3, 640, 640)).cuda(), ),
        # (torch.zeros((1, 12, 320, 320)).cuda(), ),                          # 当不导出Focus层时，输入需要修改
        "yolov5s-5.onnx",                                                   # 导出onnx文件名称
        export_params=True,                                                 # 导出参数
        training=torch.onnx.TrainingMode.TRAINING,                              # 推理模式导出,会执行BN合并
        verbose=False,                                                      # 是否打印转换的细节信息
        input_names=["input"],                                              # 指定输入的名字
        output_names=["p8", "p16", "p32"],                                  # 指定输出的名字
        dynamic_axes={
            "input": {0: "batch_size"},
            "p8": {0: "batch_size"},
            "p16": {0: "batch_size"},
            "p32": {0: "batch_size"}
        },                                                                  # 动态Batch导出
        opset_version=11,                                                   # 算子版本
        do_constant_folding=True,                                           # 默认是true   
        enable_onnx_checker=True                                            # 导出时是否检查算子，如果存在自定义算子，必须设置为True才能正确导出 
    )






if __name__ == "__main__":

    from utils.general import setup_random_seed
    import cv2
    import torch
    

    setup_random_seed(3)

    model = YoloV5("/workspace/yolov5-pro/models/yolov5s-v2.yaml").cuda()
    print(model.anchors)
    # model.train()
    # image = cv2.imread("/workspace/yolov5-pro/data/images/bus.jpg")
    # image = cv2.resize(image, dsize=(640, 640))
    # image = image.transpose(2, 0, 1)
    # # print(image.shape)
    # input_tensor = torch.as_tensor(image[None], dtype=torch.float32).cuda()
    # print(input_tensor.shape)


    # # x1, x2, x3 = model(input_tensor, False)
    # # # # a = [eval_strings(item) for item in [None, 2, "nearest"]]
    # # # # print(type(a))
    # # print(x1.shape, x2.shape, x3.shape)

    # inference_output = model(input_tensor, False)

    # # print(inference_output.shape)
    # print(inference_output[0].shape, inference_output[1].shape, inference_output[2].shape)
    
    
    # model.fuse()
    
    
    export_onnx(model)
    
    










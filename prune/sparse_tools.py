#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sparse_tools.py
@Time    :   2023/01/09 10:35:39
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   yolov5稀疏训练, 针对部分BN层权重执行L1正则, Bottleneck不进行稀疏化
'''


import torch
import torch.nn as nn   

from models.common import Bottleneck






# 对BN层执行L1正则，优化BN层参数,实现稀疏化训练
class SparseBN:
    def __init__(self, model, sr=0.001, epochs=100):
        self.model = model                                          # 网络模型
        self.sr = sr                                                # 稀疏化系数,也就是L1正则化的系数
        self.epochs = epochs                                        # 训练轮数,用于动态改变BN正则化系数
        self.ignore_bn_list = self.acquire_ignore_bn_list()         # 不需要稀疏化的BN层列表


    # 为L1正则化的BN层, 添加相应梯度
    def add_bn_grad(self, epoch):
        
        # sr_weight = self.sr * (1 - 0.9 * epoch / self.epochs)       # BN权重参数的L1正则化系数
        sr_weight = self.sr
        sr_bias = self.sr * 10                                      # BN偏置参数的L1正则化系数,它比权重系数大,是为了更快趋于零
        
        # 为什么对BN权重参数使用L1正则？因为它可以稀疏化参数,使得绝大多数参数趋于零
        # 当BN权重趋于零时,对应的卷积对模型的贡献较小,可以裁减掉
        # L1损失函数是绝对值,它的导数是 -1、0、1
        # 因此, 每一个参数加入L1正则之后, 它们的梯度是在原有梯度基础上加上一个数值(sr * sign(BN.weight.data))
        for k, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and (k not in self.ignore_bn_list):            # 如果BN层需要正则化
                m.weight.grad.data.add_(sr_weight * torch.sign(m.weight.data))              # 针对BN权重, 添加一个梯度
                m.bias.grad.data.add_(sr_bias * torch.sign(m.bias.data))                    # 针对BN偏置, 添加一个梯度
                pass
            pass
        
        pass


    # 获取不正则化的BN层列表
    def acquire_ignore_bn_list(self):
        # 存在瓶颈层的地方不进行通道剪枝,例如Bottleneck模块,不进行剪枝
        # 跳跃层剪枝之后,不能确保输入输出通道数一致
        # 这里是针对C3卷积模块的处理策略,其它自定义模块采用类似操作
        ignore_bn_list = []
        for k, m in self.model.named_modules():
            if isinstance(m, Bottleneck):
                if m.add:                                                       # Bottleneck存在跳跃链接
                    ignore_bn_list.append(k.rsplit(".", 2)[0] + "cv1.bn")       # C3网路层中的cv1卷积后的BN不进行正则化
                    ignore_bn_list.append(k + ".cv1.bn")                        # Bottleneck中的cv1卷积的BN
                    ignore_bn_list.append(k + ".cv2.bn")                        # Bottleneck中的cv1卷积的BN
           
        return ignore_bn_list


    # BN权重与偏置的分布
    def bn_distribution(self, abs=True, sort=True):
        module_weight_list = []                                                             # 存储每一个BN的权重值,也就是gamma的值
        module_bias_list = []                                                               # 存储每一个BN的偏置的值,也就是beta
        for k, m in self.model.named_modules():                                             # 遍历每一个网络层
            if isinstance(m, nn.BatchNorm2d) and (k not in self.ignore_bn_list):            # 如果当前层需要剪枝
                bn_w = m.state_dict()["weight"]                                             # 获取BN层权重的值
                bn_b = m.state_dict()["bias"]                                               # 获取BN层的偏置的值
                module_weight_list.append(bn_w)                                             # 将每一层的BN权重添加到列表
                module_bias_list.append(bn_b)                                               # 将每一层的BN偏置添加到列表
        
        index = 0
        size_list = [item.data.shape[0] for item in module_weight_list]                     # 确定每一层的BN权重的维度
        bn_weights = torch.zeros(sum(size_list))                                            # 创建一个大的tensor,维度是所有BN权重维度的展开
        bn_biases  = torch.zeros(sum(size_list))                                            # 创建一个大的tensor,维度是所有BN权重维度的展开
        for i, size in enumerate(size_list):                                                # 遍历所有的BN层列表
            if abs:
                bn_weights[index:(index + size)] = module_weight_list[i].data.abs().clone() # 获取BN权重的值,并重新克隆赋值
                bn_biases[index:(index + size)]  = module_bias_list[i].data.abs().clone()   # 获取BN偏置的值,并重新克隆赋值
            else:
                bn_weights[index:(index + size)] = module_weight_list[i].data.clone()       # 获取BN权重的值,并重新克隆赋值
                bn_biases[index:(index + size)]  = module_bias_list[i].data.clone()         # 获取BN偏置的值,并重新克隆赋值
            index += size
        
        if sort:
            bn_weights = torch.sort(bn_weights)[0]
            bn_biases = torch.sort(bn_biases)[0]
            
        bn_weights = bn_weights.cpu()
        bn_biases  = bn_biases.cpu()
        

        return bn_weights, bn_biases









































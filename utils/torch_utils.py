#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   torch_utils.py
@Time    :   2022/11/10 14:13:17
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   torch 相关的工具
'''

import math
import torch
import torch.nn as nn
from copy import deepcopy



# 模型权重初始化, 针对偏置、检测头,可根据任务指定特殊的初始化
def initialize_weights(model):
    for m in model.modules():
        m_type = type(m)
        if m_type is nn.Conv2d:                     # 其实默认就是凯明初始化
            pass                                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif m_type is nn.BatchNorm2d:
            m.eps = 1e-3                            # 默认是 1e-5, 分母中添加一个值,为了计算稳定性
            m.momentmum = 0.03                      # 默认是 0.1, 一个运行过程中均值和方差的一个估计参数
            nn.init.normal_(m.weight.data, mean=0, std=1)                           # 初始化为标准正态分布
            nn.init.constant_(m.bias.data, 0)
        elif m_type in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
    pass



# 合并卷积层和批归一化层
def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    # 卷积核参数的shape是 [out_channels, in_channels, kernel_size, kernel_size]=[128, 64, 3, 3]
    # 重新定义一个卷积层
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True).requires_grad_(False).to(conv.weight.device)
    
    # 准备滤波器参数
    # 这里假设 卷积核是 128个3*3*64
    # 128是输出通道数, 64是输入通道数, 3*3是卷积核大小
    w_conv = conv.weight.clone().view(conv.out_channels, -1)                        # shape=[128, -1]
    
    # 计算出每个通道对应的 gamma / sqrt(running_var + eps),共128个，并转换成对角线，变成128*128
    # 这样可以对每一个通道的卷积核每一个元素进行变换
    scale_factor = bn.weight.div(torch.sqrt(bn.running_var + bn.eps))
    w_bn = torch.diag(scale_factor)           # 维度是[128, 128]
    
    # 这一步就得到了融合后的卷积核的权重，gamma * w_conv / (running_var + eps)
    # 将新的权重复制到新的卷积中
    # fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))                         # yolov5的写法，利用矩阵乘法
    fused_conv.weight.copy_((scale_factor.view(-1, 1) * w_conv).view(fused_conv.weight.shape))              # 自己的写法，利用广播机制
    
    
    # 准备偏置参数
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias            # shpe=[128]
    new_bias = scale_factor * (b_conv - bn.running_mean) + bn.bias                                                      # 自己写法，未采用yolov5写法
    fused_conv.bias.copy_(new_bias)
    
    return fused_conv


# 参数分组优化
def parameters_group(model):
    """
        # 返回值参数：
            1. pg0 是指卷积层的weight,yolov5只对卷积层weight进行权重衰减
            2. pg1 是指BN层的weight
            3. pg2 是指所有网络层的bias
    
    """
    
    # 将优化器参数分为3个小组，分别设置不同的优化器参数
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_parameters():
        if ".bias" in k:                                # 这里包括卷积核BN层的bias
            pg2.append(v)
        elif "weight" in k and ".bn" not in k:          # 这里是指卷积层的weight,不包括BN层的weight
            pg0.append(v)
        else:
            pg1.append(v)                               # 这里其实就是BN层的weight
            
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    return pg0, pg1, pg2



# 学习率调度器, 学习率策略
def acquire_lr_scheduler(optimizer, lrf=0.001, epochs=100, T=100, cos_lr=True, save_dir=""):
    """
        ### 学习率公式：
            1. 余弦退火学习率, eta_min + (1/2) *(eta_max-eta_min) * (1 + cos(x * pi / epochs))
            2. 设置周期性重启,周期性重启的周期是50个epoch
            3. lrf,表示学习率的最终值，通过超参数设置
            4. cos_lr, 选择余弦学习率,还是线性学习率
    """
    if cos_lr:
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * (1 - lrf) + lrf                  # 余弦退火学习率 
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf                                                   # 线性学习效率
    # lf = lambda x: (((1 + math.cos((x % 50) * math.pi / 50)) / 2) ** 1.0) * 0.8 + 0.2               # 余弦退火 + 周期性重启
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf



# 智能优化器, 针对不同参数进行分组优化
def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    """
        ### 函数说明：
            1. 对不同参数执行分组优化
            2. 优化器的选择, Adam AdamW RMSProp SGD
        ### 分组优化
            1. weights with decay
            2. weights no decay
            3. biases no decay
    """
    pg0, pg1, pg2 = parameters_group(model)                                                             # 参数分组
    
    if name == "Adam":
        optimizer = torch.optim.Adam(pg2, lr=lr, betas=(momentum, 0.999))                               # adam优化，BN权重不衰减
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(pg2, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(pg2, lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(pg2, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')
    
    optimizer.add_param_group({"params": pg0, "weight_decay": decay})                                    # 卷积权重执行权重衰减
    optimizer.add_param_group({"params": pg1, "weight_decay": 0.0})                                      # BatchNorm2d weights不进行权重衰减
    del pg0, pg1, pg2

    return optimizer



# 智能恢复训练
def smart_resume(model, optimizer, weights="yolov5s.pt", results_file="results.txt", device=None, ema=None, epochs=300):
    """
        ### 函数说明
            1. 根据检查点文件,恢复模型训练
            2. 如果检查点中存在训练结果,也恢复保存起来
            3. 需要提供检查点文件
        ### 参数说明
            1. model, 模型的网络结构
            2. optimizer, 用于恢复优化策略以及其参数
            3. weights, 用于恢复模型的权重文件
            4. results file, 保存训练结果的文件
            5. device, 设备
            6. ema, 指数滑动平均
            7. epochs, 完整训练的轮数
        ### 返回值
            1. start epoch, 恢复训练的起始轮数
            2. best map, 之前训练的最好的mAP值
    
    """
    start_epoch, best_map = 0, 0.0
    
    assert weights.endswith(".pt") == True                                                  # 断言,确保权重文件正确
    checkpoint = torch.load(weights, map_location=device)                                   # 加载检查点文件到指定设备
    
    # 开始逐步恢复
    model.load_state_dict(checkpoint["model"])
    if checkpoint["optimizer"] is not None:                                                                 # 如果检查点文件中包括优化器参数
        optimizer.load_state_dict(checkpoint["optimizer"])                                                  # 加载优化器参数
    
    if ema and checkpoint.get("ema"):                                                                       # EMA（Exponential Moving Average）是指数移动平均值
        ema.ema.load_state_dict(checkpoint["ema"].float().state_dict())
        ema.updates = checkpoint["updates"]
        
    if checkpoint.get("training_results"):                                                                  # 如果训练结果存在
        with open(results_file, "w") as f:                                                                  # 保存训练结果到本地
            f.write(checkpoint["training_results"])
            
    if checkpoint["epoch"] is not None:                                                                     # 确定恢复训练的起始轮数
        start_epoch = checkpoint["epoch"] + 1
        if epochs < start_epoch:
            print("%s has been trained for %g epochs. Resume train util %g epochs." %(weights, start_epoch, epochs))
        else:
            print("%s has been trained for %g epochs. Resume train util %g epochs. train finished!!!" %(weights, start_epoch, epochs))
            return None
    if checkpoint.get("best_map"):
        best_map = checkpoint["best_map"]
        
    return start_epoch, best_map


# 加载预训练权重,注意其与恢复训练的区别
def load_pretrained_weights(model, weights="weights/yolov5s.pt", device=None):
    """
        ### 函数说明
            1. 将提供的权重文件,加载到模型中
            2. 注意anchor这类与场景相关的参数,不需要加载
            3. 该操作与模型恢复是不一样的,切记！！！
    
    """
    assert weights.endswith(".pt") == True                                  # 断言,确保权重文件正确
    
    if weights.endswith(".pt"):                                             # pytorch格式的权重文件
        
        print("1111111111111111111")
        checkpoint = torch.load(weights, map_location="cpu")               # 加载检查点文件
        print("222222222222222222222222")
        
        try:
            exclude = ["anchor"]                                            # 需要排除的keys, 每个任务的anchor是不同的,因此不需要加载
            ckpt_model = dict()
            for k, v in checkpoint["model"].float().state_dict().items():
                if k in model.state_dict() and not any(x in k for x in exclude) and model.state_dict()[k].shape == v.shape:        # 根据key值, 筛选权重
                    ckpt_model[k] = v
            # 加载权重到模型
            model.load_state_dict(ckpt_model, strict=False)
            print('Transferred %g/%g items from %s' % (len(ckpt_model), len(model.state_dict()), weights))
        except KeyError as e:
            s = "%s is not compatible with model. This may be due to model differences or %s may be out of date. " \
                "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                % (weights, weights, weights)
            raise KeyError(s) from e
        





# 模型的指数滑动平均
class ModelEMA:
    """ 
        ### 功能说明
            1. 参考
                Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
                Keeps a moving average of everything in the model state_dict (parameters and buffers)
                For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            2. 指数移动平均: 是以指数式递减加权的移动平均。各数值的加权影响力随时间而指数式递减,越近期的数据权重影响力
               越重,但较旧的数据也给予一定的加权值。
            3. 优点是: 与一般的加权平均相比,使用指数移动平均的好处是,不需要保存前面所有时刻的实际数值,并且在计算vt的过程中
                       是逐步覆盖的,因此可以减少内存占用
            4. 训练时候,维护影子变量,但采用原始权重做梯度更新; 推理时采用影子变量替换原始权重
            5. 影子变量的更新更加平滑,在测试数据上效果更好、更健壮
            6. 当decay=0.999时,表示在最后的 1000 次训练过程中，模型早已经训练完成，正处于抖动阶段，而滑动平均相当于将最后
               的 1000 次抖动进行了平均，这样得到的权重会更加 robust
    """
    # 最开始 decay为0,慢慢变大并趋近于0.9999,因此，刚开始迭代时，我们更相信当前的模型权重，较少的相信之前的EMA权重
    # 随着迭代的进行，我们开始有理由相信之前的EMA权重有一定的可信度
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # 创建EMA
        self.ema = deepcopy(model).eval()                           # FP32 EMA
        self.updates = updates                                      # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))     # decay exponential ramp (to help early epochs)
        for param in self.ema.parameters():
            param.requires_grad_(False)
        
    
    # 更新的公式是 v_t = beta * v_t-1 + (1 - beta) * theta_t
    def update(self, model):
        # update EMA parameters
        with torch.no_grad():                                           # 这里只是更新EMA权重，梯度更新仍然是针对当前模型的
            self.updates += 1
            beta = self.decay(self.updates)
            
            msd = model.state_dict()                                    # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:                           # true for FP16 and FP32
                    v *= beta
                    v += (1 - beta) * msd[k].detach()
            
        
        
        
        
        
        
        
        








if __name__ == "__main__":
    
    # a = torch.tensor([1, 2, 3])
    # a = torch.diag(a)
    # print(a)
    
    import sys
    import torch
    sys.path.append("..")
    
    from models.yolo import YoloV5
    from utils.plots import plot_lr_scheduler
    
    epochs = 300
    
    
    model = YoloV5("/workspace/yolov5-pro/models/yolov5s-v2.yaml")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.90, 0.999))
    
    scheduler = acquire_lr_scheduler(optimizer, epochs)
    plot_lr_scheduler(optimizer, scheduler, epochs, "./")

    # parameters_group(model)


















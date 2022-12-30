import torch
import torch.nn as nn

from models.common import Bottleneck
from visdom import Visdom as vis


# 针对C3,也就是yolov5-6.0版本的剪枝
def optimizer_BN(model, sr=0.0001, epoch=0, epochs=100):
    
    # sr被称为稀疏率(sparsity rate),我觉得就是L1正则化系数
    srtmp = sr * (1 - 0.9 * epoch / epochs)                         # BN.weight的系数

    ignore_bn_list = []
    
    for k, m in model.named_modules():
        if isinstance(m, Bottleneck):
            # 存在瓶颈层的这一块是不剪枝的
            # 因为剪枝之后不能保证输入输出通道数一致
            # 这里是针对C3层的处理策略,其它网络类似
            # 这里所说的C3是带跳跃链接的C3
            if m.add:                                                           # 存在跳跃链接
                ignore_bn_list.append(k.rsplit(".", 2)[0] + "cv1.bn")              # C3网络层中的cv1卷积的BN不正则化
                ignore_bn_list.append(k + ".cv1.bn")                            # BottleNect中的cv1卷积的BN
                ignore_bn_list.append(k + ".cv2.bn")                            # BottleNect中的cv2卷积的BN
        if isinstance(m, nn.BatchNorm2d) and (k not in ignore_bn_list):
            
            # 为什么对BN参数使用L1正则？因为它可以稀疏化参数,使参数趋于零
            # L1损失函数是绝对值,它的倒数是-1、0、1
            # 因此每一个参数加入L1正则之后的梯度是在原有梯度基础上加上一个数值(sr * sign(BN.weight.data))
            m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))                      # L1 正则,直接处理梯度      
            m.bias.grad.data.add_(sr * 10 * torch.sign(m.bias.data))                        # L1 正则,这个可以不需要
            
        
            pass
        # print(m.state_dict())

    return ignore_bn_list



# 显示BN权重
def show_BN_hist(model, ignore_bn_list):
    module_weight_list = []
    module_bias_list = []
    for k, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and (k not in ignore_bn_list):
            bn_w = m.state_dict()["weight"]
            bn_b = m.state_dict()["bias"]
            module_weight_list.append(bn_w)
            module_bias_list.append(bn_b)
            
    size_list = [item.data.shape[0] for item in module_weight_list]

    bn_weights = torch.zeros(sum(size_list))
    bn_bias    = torch.zeros(sum(size_list))
    index = 0
    for i, size in enumerate(size_list):
        bn_weights[index:(index + size)] = module_weight_list[i].data.abs().clone()
        bn_bias[index:(index + size)]    = module_bias_list[i].data.abs().clone()
        
        index += size
        
    bn_weights = bn_weights.cpu().tolist()
    bn_bias = bn_bias.cpu().tolist()
    bn_weights.sort()
    bn_bias.sort()
    
    
    return bn_weights, bn_bias

if __name__ == "__main__":
    
    from models.yolo import YoloV5
    
    model = YoloV5(yaml_cfg_file="/workspace/yolov5-pro/models/yolov5s.yaml").cuda()
    model.train()
    
    
    ignore_bn_list = optimizer_BN(model)

    show_BN_hist(model, ignore_bn_list)














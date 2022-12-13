#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   yolov5_loss.py
@Time    :   2022/12/01 11:32:40
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   None
'''

import os
import sys
import torch
import torch.nn as nn

sys.path.append("..")

from utils.metrics import BboxIoU




def smooth_BCE(eps=0.1):
    '''
        ### 函数说明：
            1. 标签平滑, Label smoothing, 出自 https://arxiv.org/pdf/1902.04103.pdf eqn 3
            2. 这段代码出自 https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
            3. 这里是针对二分类的
        ### 返回值
            1. 给予正样本的概率值
            2. 给予负样本的概率值
    '''
    return 1 - 0.5 * eps, 0.5 * eps


class FocalLoss:
    def __init__(self, loss_func, alpha=0.25, gamma=1.5):
        pass



# 重写 loss,为了排除错误
class ComputeLoss:
    
    def __init__(self, model, method="CIoU", autobalance=False):
        self.device = next(model.parameters()).device                               # 得到模型所在的设备
        self.hyp    = model.hyp                                                     # 模型超参数
        self.method = method                                                        # box loss 的计算方法
        
        # 定义类别、置信度的损失函数
        self.cls_BCE_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp.get("cls_pw", 1.0)], device=self.device))        # 类别损失函数,采用二元交叉熵损失,并为正样本设置合适的权重
        self.obj_BCE_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp.get("obj_pw", 1.0)], device=self.device))        # 置信度损失函数,采用二元交叉熵损失,并为正样本设置合适的权重
        self.box_IoU_func = BboxIoU(method=self.method, eps=1e-7)                                                                   # 计算各种IOU的值,然后再计算损失
        
        # 是否采用Focal Loss
        fl_alpha = self.hyp.get("fl_alpha", 0.25)                                                                                   # 超参数列表中设置FocalLoss的alpha参数
        fl_gamma = self.hyp.get("fl_gamma", 0.0)                                                                                    # 超参数列表中设置FocalLoss的gamma参数,大于0才采用FocalLoss计算
        if fl_gamma > 0:
            self.cls_BCE_func = FocalLoss(self.cls_BCE_func, alpha=fl_alpha, gamma=fl_gamma)                                        # 针对类别,采用Focal Loss
            self.obj_BCE_func = FocalLoss(self.obj_BCE_func, alpha=fl_alpha, gamma=fl_gamma)                                        # 针对置信度,采用Focal Loss
        
        # 类别标签平滑
        self.cp, self.cn = smooth_BCE(eps=self.hyp.get("label_smoothing", 0.0))                                                     # 根据超参数,对类别进行label smoothing
        
        # 获取检测器头部的各个参数
        # 包括检测层的数目、类别数、每一个检测层anchor的数目、anchors和strides
        detect_m     = model.model[-1]                                                                                              # 网络的Detect层
        self.nl      = detect_m.num_layer                                                                                           # 检测层数目,默认情况下是3
        self.na      = detect_m.num_anchor_per_level
        self.nc      = detect_m.num_cls
        self.anchors = detect_m.anchors                                                                                             # anchors,这里是指特征图尺度下的anchors,已经除了stride
        self.strides = detect_m.strides                                                                                             # 每一个检测头的下采样倍数
        
        # 平衡参数,这里是指针对置信度损失,平衡各个检测层的贡献
        self.autobalance = autobalance                                                                                              # 不同检测头的权重才采取不同的平衡策略
        self.balance     = {3: [4.0, 1.0, 0.4], 4: [4.0, 1.0, 0.4, 0.1], 5: [4.0, 1.0, 0.25, 0.06, 0.02]}                           # 不同检测层置信度损失的平衡系数
        self.balance     = self.balance.get(self.nl, [4.0, 1.0, 0.4])   
        self.ssi         = list(self.strides).index(16) if autobalance else False
        
        # 暂时不知道干什么用
        self.gr = 1.0
        
    def __call__(self, predictions, targets):
        lcls = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)                                                        # 根据目标和anchors的匹配情况，制作的标签，包括了扩充样本,tbox中的xy就是需要回归的小数,取值范围是(-0.5, 1.5)
        
        # 计算loss
        for i, predict in enumerate(predictions):
            predict = predict.view(-1, self.na, 5 + self.nc, predict.shape[2], predict.shape[3])                            # 转换维度到[b, 3, 25, 80, 80]       
            predict = predict.permute(0, 1, 3, 4, 2).contiguous()                                                           # 转置,维度变换为[b, 3, 80, 80, 25]
            img_id, anc_id, grid_y, grid_x = indices[i]
            tobj = torch.zeros(predict.shape[:4], dtype=predict.dtype, device=self.device)                                  # 用来填充targets objectness,制作类别置信度标签,维度是[b, 3, 80, 80]
            
            # print(tobj.shape)
            
            n = img_id.shape[0]                 # 匹配上的目标数量
            if n:
                # pt_xy的维度是[n, 2]
                # pt_wh的维度是[n, 2]
                # pt_cls的维度是[n, 20]
                pt_xy, pt_wh, _, pt_cls = predict[img_id, anc_id, grid_y, grid_x].split((2, 2, 1, self.nc), 1)
                
                # 边界框回归损失
                pt_xy = pt_xy.sigmoid() * 2 - 0.5                       # 范围是(-0.5, 1.5)
                pt_wh = (pt_wh.sigmoid() * 2) ** 2 * anchors[i]         # 特征图尺度下的预测框, 范围是(0, 4*anchors)
                pt_box = torch.cat((pt_xy, pt_wh), dim=1)               # 预测框
                iou = self.box_IoU_func(pt_box, tbox[i])                # 计算各种IoU的值
                lbox += (1 - iou).mean()                                # 添加box回归损失
                # print("iou: ", iou)
                
                # 置信度,objectness,当存在目标时，置信度设置为IoU的值
                iou = iou.detach().clamp(0, 1).type(tobj.dtype)         # 将IoU分离处理，不在计算图中,不计算梯度
                tobj[img_id, anc_id, grid_y, grid_x] = iou              # 用IoU的值作为有无目标的置信度,它既可以表示有无目标,还可以表示存在目标的可信度有多大
                
                # classification
                if self.nc > 1:                                                                 # 类别为1时,不计算分类损失,只计算置信度损失
                    gt_cls = torch.full_like(pt_cls, self.cn, device=self.device)               # gt_cls的维度是[n, 20], 默认值设置为标签平滑后的负样本的概率
                    gt_cls[range(n), tcls[i]] = self.cp                                         # 设置正样本的概率,这里不一定是1,它是一个接近于1的数值
                    lcls += self.cls_BCE_func(pt_cls, gt_cls)                                   # 分类损失,二元交叉熵

            # 置信度损失,无论是否存在目标,都应该计算置信度损失,它表示当前网格存在目标的置信度
            obj_i= self.obj_BCE_func(predict[..., 4], tobj)                                     # 第i层的obj_loss
            lobj += obj_i * self.balance[i]                                                     # 计算总的obj_loss,这里需要平衡每一层
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obj_i.detach().item()
            
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]
        
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    
    


    
    # 根据匹配策略,构建标签,这里我们采取每一个检测层构建一个匹配标签
    def build_targets(self, predictions, targets):
        na, nt = self.na, targets.shape[0]                                      # anchor的数目、gt标签的数目
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)                                # 归一化目标框到gridspace的增益系数
        ai = torch.arange(na, device=self.device).view(na, 1).repeat(1, nt)      # repeat表示在行维度重复1次，列的维度重复4次，从[3,1]->[3,nt],用来表示anchor的索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)       # 最终的维度是[3, nt, 7],最后一个维度对应的是anchor的索引
        # 到这一步targets的维度是[3, nt, 7]
        # 等于是将原始的targets复制了3分，并且每一份在最后一个维度添加一个anchor索引
        # 原始的targets的维度是[nt, 6], [img_id, cls_id, x, y, w, h]
        # 转化后的targets的维度是[3, nt, 7],其中7这个维度是[img_id, cls_id, x, y, w, h, anchor_idx]
        
        g = 0.5
        # 我们通常以方框左上角表示grid_x, grid_y
        # 定义中心位置, 以及上下边界
        cell_center = 0.5                                                         # 表示一个像素的中心点, 根据判断在中心那边增加样本
        
    
        
        offset_boundary = torch.FloatTensor([
            [ 0,  0],                                   # 当前网格
            [+1,  0],                                   # 右边扩充
            [ 0, +1],                                   # 下边扩充
            [-1,  0],                                   # 左边扩充
            [ 0, -1]]).to(self.device)                  # 上边扩充
        
        for i, predict in enumerate(predictions):                                  # 遍历每一个检测层
            anchors, shape = self.anchors[i], predict.shape                     # 当前层对应的anchor、当前层特征的shape NCHW
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]                       # xywh的增益
            
            # 匹配targets到anchors
            gained_targets = targets * gain                                     # 特征图尺度下的gt标签, 维度是[3, n, 7]
            
            if nt > 0:
                # 过滤匹配targets
                wh_ratio = gained_targets[..., 4:6] / anchors[:, None]                              # anchors本来维度是[3, 2], 转变为[3, 1, 2], 结果的维度是[3, n, 2]
                j = torch.max(wh_ratio, 1 / wh_ratio).max(dim=2)[0] < self.hyp['anchor_t']          # 维度是 [3, n]
                select_targets = gained_targets[j]                                                  # 维度是[m, 7],m表示匹配上的目标数目
                
                # offsets
                gt_xy = select_targets[:, 2:4]                                        # 目标框的中心点坐标,带有小数,维度是[m, 2]
                
                low_boundary = 1.0                                                  # 当cx > 1, 左边才可以扩充, cy > 1, 上边才可以扩充
                high_boundary = gain[[2, 3]] - 1.0                                  # 当cx < w-1, 右边才可以扩充, cy < h-1, 下面才可以扩充
                
                expand_left_mask, expand_top_mask     = ((gt_xy % 1 < cell_center) & (gt_xy > low_boundary)).T       # 表示左边和上边可以扩充,结果的维度是[2, m]
                expand_right_mask, expand_bottom_mask = ((gt_xy % 1 > cell_center) & (gt_xy < high_boundary)).T      # 表示右边和下边可以扩充,结果的维度是[2, m]
                
                all_sample_mask = torch.stack((torch.ones_like(expand_left_mask), expand_left_mask, expand_top_mask, expand_right_mask, expand_bottom_mask))        # 维度是[5, m]
                
                select_targets = select_targets.repeat(5, 1, 1)[all_sample_mask]                                # 初始的select_targets维度是[5, m, 7],最终结果维度大概是[3m, 7]
               
                # print(all_sample_mask.shape)                                  # 维度是[5,m]
                # print(select_targets.shape)                                   # 维度是[3m, 7]
                
                 # 原始匹配的目标不需要偏移量
                 # 维度是 [1. m, 2] + [5, 1, 2]  -> [5, m, 2]
                xy_offsets = (torch.zeros_like(gt_xy)[None] + offset_boundary[:, None])[all_sample_mask]
                
                # print(xy_offsets.shape)
            
            else:
                select_targets = targets[0]
                xy_offsets = 0
            
            # 定义
            bc, gt_xy, gt_wh, a = select_targets.chunk(4, 1)                    # 本身维度是[m, 7],在第1个维度切分为4份
            anc_id, (img_id, cls_id) = a.long().view(-1), bc.long().T
            
            grid_xy = (gt_xy - xy_offsets).long()
            grid_x, grid_y = grid_xy.T
            
            # 添加
            indices.append((img_id, anc_id, grid_y.clamp(0, shape[2] - 1), grid_x.clamp(0, shape[3] - 1)))  # img_id, anc_id, grid_y, grid_x
            tbox.append(torch.cat((gt_xy - grid_xy, gt_wh), dim=1))                                         # 维度是 3个[3m, 4]
            anch.append(anchors[anc_id])
            tcls.append(cls_id)
            
            # print(indices[0].shape)
            # print(tbox[0].shape)
            # print(anch[0].shape)
            # print(tcls[0].shape)
            # break
        
        
        return tcls, tbox, indices, anch









if __name__ == "__main__":
    
    from models.yolo import YoloV5
    hyp = {"anchor_t": 4.0, "box": 0.05, "cls": 0.01, "obj": 0.02, "label_smoothing": 0}
    device = "cuda"
    
    model = YoloV5(yaml_cfg_file="/workspace/yolov5-pro/models/yolov5s-v2.yaml")
    model.hyp = hyp
    
    model.to(device)
    model.train()
    
    input = torch.zeros((8, 3, 640, 640)).float().to(device)
    
    model(input)
    
    loss_func = ComputeLoss(model, autobalance=True)
    
    
    predicts_0 = torch.ones((8, 75, 80, 80), dtype=torch.float32).to(device)
    predicts_1 = torch.ones((8, 75, 40, 40), dtype=torch.float32).to(device)
    predicts_2 = torch.ones((8, 75, 20, 20), dtype=torch.float32).to(device)
    predicts = [predicts_0, predicts_1, predicts_2]
    targets = torch.ones((10, 6), dtype=torch.float32).to(device)
    targets[:, 2:]  = targets[:, 2: ] * 0.09
    
    tloss, llist = loss_func(predicts, targets)
    
    print("tloss: ", tloss)
    print(llist)
    # print(torch.tensor([2, 3, 4, 5])[[3, 2, 3, 2]])
    
    # ai = torch.arange(3, device="cuda").float().view(3, 1).repeat(1, 4)

    # print(ai)
    # print(ai[..., None])
    # print(ai[..., None].shape)






















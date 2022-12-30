#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss_yolov5.py
@Time    :   2022/10/12 15:27:23
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   yolov5 loss calculate method
'''

import os
import sys
import torch
import platform
import numpy as np
import torch.nn as nn

from pathlib import Path


sys.path.append("..")


from utils.metrics import BboxIoU


"""
    # 在yolov5-6.2版本中, 我发现box损失采用的是 CIoU损失, 好像之前的版本采用的是GIoU
    # 针对ground truth的objectness, 概率值仍然设置为边界框回归中计算出来的CIOU, 也就是采用哪种损失就设置为谁的值, 需要裁剪到[0,1]

"""





class ComputeLoss(nn.Module):
    def __init__(self, num_classes=20, anchors=[], hyp=None, device="cuda", box_reg="SIoU"):
        super().__init__()
        self.device = device                                                    # 确定计算loss在那个设备上
        self.hyp = hyp                                                          # 超参数
        
        # 只能3个或者4个预测层
        self.strides = [8, 16, 32] if len(anchors) == 3 else [4, 8, 16, 32]     # 预测层下采样的倍数
        self.num_classes = num_classes                                          # 检测目标的类别数
        self.anchor_t = self.hyp["anchor_t"]                                    # 分配targets到哪一个预测层的阈值
        
        # print(anchors)
        # print(torch.tensor(anchors).shape)

        self.anchors = (torch.tensor(anchors).view(len(anchors), 3, 2) / torch.FloatTensor(self.strides).view(len(anchors), 1, 1)).to(self.device)

        self.offset_boundary = torch.FloatTensor([
            [+1, 0],
            [0, +1],
            [-1, 0],
            [0, -1]]).to(self.device)                                                               # 扩充样本时使用

        self.num_anchor_per_level = self.anchors.size(1)                                            # 每一个预测层有3个anchor
        
        self.BCE_objectness = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp.get("cls_pw", 1.0)], device=self.device))                                # 二元交叉熵损失
        self.BCE_classification = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp.get('cls_pw', 1.0)], device=self.device))                            # 多个二元交叉熵损失
        self.balance = [4.0, 1.0, 0.4] if len(anchors) == 3 else [4.0, 1.0, 0.4, 0.1]               # 针对三个尺度上的 objectness损失的平衡策略 8, 16, 32
        
        self.IoU_box_regression = BboxIoU(method=box_reg, eps=1e-7)




    def forward(self, predicts, targets):
        """
            ### 参数
                - predicts: [N, C=3, H, W, (5+cls)]
                - targets:  [M, 6], 分别表示 [img_id, class_id, cx, cy, w, h], 坐标是归一化后的
        """
        self.num_target = targets.size(0)                                                    # 当前批次的图像一共包含多少个目标
        self.matched_num_per_target = torch.zeros(self.num_target).to(self.device)           # 记录每个目标的匹配情况,最终统计3个阶段都没匹配上的目标,未计算扩充的样本
        self.num_positive_sample = torch.tensor(0).to(self.device)                           # 记录当前批次, 正样本的数目, 包括扩充之后，计算总的匹配数目

        self.loss_box_regression = torch.FloatTensor([0]).to(self.device)                    # box回归的loss
        self.loss_classification = torch.FloatTensor([0]).to(self.device)                    # 类别损失loss
        self.loss_objectness     = torch.FloatTensor([0]).to(self.device)                    # objectness loss



        for ilevel, cur_level_predict in enumerate(predicts):                           # 遍历每一个预测层, 一共有3个预测层
            feature_map_w, feature_map_h = cur_level_predict.shape[2:4]                 # 当前预测层的特征图宽高尺寸
            
            # 根据匹配策略, 匹配标签, yolov5还对标签进行了扩充
            select_targets, select_anchor_index, xy_offset = self.build_targets(targets, ilevel, feature_map_w, feature_map_h)

            expanded_num_matched_target = len(select_targets)                                # 当前预测层扩充样本后, 与anchor匹配的总数目
            self.num_positive_sample += expanded_num_matched_target                          # 将总数目记录下来, 最终计算平均的匹配数目,每一个批次目标匹配

            # 到此为止, 我们已经选出了当前预测层匹配成功的目标以及扩充样本,并且知道其对应的anchor的索引,cx,cy的中心偏移量(用于计算扩充样本的 grid_x, grid_y)

            # 准备计算loss,计算结果保存起来, 并没有传递
            self.compute_loss(cur_level_predict, select_targets, select_anchor_index, xy_offset, ilevel)

        # 开始计算总的loss, 这里每种loss都有自己的权重
        self.loss_box_regression *= self.hyp["box"]                                 # 加权边界框回归损失
        self.loss_classification *= self.hyp["cls"]                                 # 加权分类损失
        self.loss_objectness     *= self.hyp["obj"]                                 # 加权目标置信度损失

        batch_size = predicts[0].shape[0]
        total_loss = (self.loss_box_regression + self.loss_classification + self.loss_objectness) * batch_size    # 总的loss
        
        matched_num_mean = (self.num_positive_sample / self.num_target).detach().cpu().item()
        unmatched_num_target = np.sum(np.where(self.matched_num_per_target.detach().cpu().numpy(), 0, 1))

        return total_loss, torch.cat((self.loss_box_regression, self.loss_classification, self.loss_objectness)).detach(), [matched_num_mean, unmatched_num_target]


    
    # 计算每一个预测层损失
    def compute_loss(self, cur_level_predict, select_targets, select_anchor_index, xy_offset, ilevel):
        """
            ### 参数
                - predicts:             网络模型的输出, 维度是 [batch, 5+num_classes, H, W]
                - select_targets:       targets与当前层anchor匹配成功的target, 包括扩充的正样本, 维度是 [N, 6], N 表示当前预测层匹配上的所有target的数目
                - select_anchor_index:  与匹配成功的target对应的anchor的索引, 维度是 [N]
                - xy_offset:            针对扩充样本,需要计算它的正确的 grid_x, grid_y, 其中select_targets内部保存的扩充样本和原始目标一样,因此通过xy_offset区别计算,维度是[N, 2]     
        """

        feature_map_w, feature_map_h = cur_level_predict.shape[2:4]                      # 当前预测层的特征图宽高尺寸
        cur_level_predict = cur_level_predict.view(-1, self.num_anchor_per_level, 5+self.num_classes, feature_map_h, feature_map_w) # [N, 3, 5+cls, H, W]
        cur_level_predict = cur_level_predict.permute(0, 1, 3, 4, 2).contiguous()        # 转变维度并连续化[N, num_anchor, H, W, 5+cls]
        feature_map_objectness = cur_level_predict[..., 4]                               # 维度是 [N, num_anchor, H, W]
        objectness_ground_truth = torch.zeros_like(feature_map_objectness)               # 维度是 [N, num_anchor, H, W]

        if len(select_targets) > 0:                                                          # 有匹配样本才会去计算Box与class的损失

            gt_img_id, gt_class_id = select_targets[:, [0, 1]].long().T                      # 获取每个目标对应的 img_id 和 class_id, 维度是 [N]
            gt_xy = select_targets[:, [2, 3]]                                                # 每个目标边界框的中心点,注意这里是feature_map尺度下的, 维度是[N, 2],带小数部分
            gt_wh = select_targets[:, [4, 5]]                                                # 每个目标边界框的宽高, 维度是[N, 2], 特征图尺度下的宽高
            grid_xy = (gt_xy - xy_offset).long()                                             # 表示每一个target中心点由那个网格来预测, 维度是[N, 2], 不带小数部分
            grid_x, grid_y = grid_xy.T                                                       # 每个目标所确定的预测网格, 维度是 [N]
            
            # 这里再次解释一下关于扩充正样本的预测网格的计算
            # 假设 cx = 1.49999, 它满足左边填充条件, grid_x的计算公式是 grid_x = (cx - 0.5).long() = 0
            # 由上面的计算可知, 左边扩充后,通过 grid_x = 0 去预测1.49999, 因此 predict的预测的x的值阈上限是 (1.49999 - 0) = 1.5(开区间)
            # 假设 cx = 1.50001, 它满足右边填充条件, grid_x的计算公式是 grid_x = (cx + 0.5).long() = 2
            # 由上面的计算可知, 右边扩充后,通过 grid_x = 2 去预测1.49999, 因此 predict的预测的x的值阈下限是 (1.50001 - 2) = -0.5(开区间)
            # 同理, gt_xy - grid_xy 的范围是(-0.5, 1.5), 这也是网络需要预测的东西
            # 因此, predict_xy的值阈也应该在这个范围, yolov5其实也是这么设计的
            gt_xy_regression_offset = gt_xy - grid_xy                                        # 网络预测所要回归的中心点坐标偏移量, 维度是 [N, 2]
            select_anchors = self.anchors[ilevel][select_anchor_index]                       # 每个target对应的anchor的宽高, 维度是 [N, 2]

            # print(grid_y, grid_x)

            # 准备predict, 在计算之前需要先转换predict
            # 因为, yolov5的预测输出的数值并不是真正的x, y, w, h, 故而并不能直接用来回归或者计算IOU
            # 下面就是predict预测的转换过程
            # 在cur_level_predict预测层中
            #       -> gt_img_id                指定 batch, 它本身的维度是 [N]
            #       -> elect_anchor_index       指定某个anchor, 它本身的维度是 [N]
            #       -> grid_y                   指定网格的height维度, 它本身的维度是 [N]
            #       -> grid_x                   指定网格的width维度, 它本身的维度是 [N]
            object_predict = cur_level_predict[gt_img_id, select_anchor_index, grid_y, grid_x]              # 得到的维度是 [N, 5+num_classes]


            # print("object_predict.shape", object_predict.shape)
            # print(cur_level_predict. is_contiguous())
            # print(cur_level_predict.shape)
            predict_xy_regression_offset = object_predict[:, [0, 1]].sigmoid() * 2.0 - 0.5                  # 使用sigmoid,以及乘以2,减去0.5是为了把值阈变为(-0.5, 1.5),维度是[N, 2]
            
            # print()


            # # 这里将预测的宽高的值阈变换为 (0, 4)倍的anchor大小
            # # 这里可以发现, 可以预测小于anchor的Bbox, 哪怕很小, 但是不会预测大于4倍anchor大小的Bbox
            # # 所以, 这里是不是可以说明, 即使我的Bbox很小, 那么我其实也可以在大特征图上预测, 但是前提是, 需要更改匹配策略, 让小目标可以匹配
            predict_wh_featuremap_scale = torch.pow(object_predict[:, [2, 3]].sigmoid() * 2.0, 2.0) * select_anchors                    # 维度是 [N, 2],

            # 开始准备计算GIOU Loss
            # 准备预测边界框与预测边界框
            gt_bboxes = torch.cat([
                gt_xy_regression_offset,
                gt_wh], dim=1)                                             # 拼接后的维度是 [N, 4], 分别是 [x, y, w, h]
            # print("gt_bboxes.shape", gt_bboxes.shape)
            predict_bboxes = torch.cat([
                predict_xy_regression_offset,
                predict_wh_featuremap_scale], dim=1)                       # 拼接后的维度是 [N, 4], 分别是 [x, y, w, h], 虽然这里的x,y是指偏移量,但是不影响直接计算,IOU计算与去掉一个grid_x, grid_y关系

            # print("predict_bboxes.shape", predict_bboxes.shape)
            giou = self.IoU_box_regression(predict_bboxes, gt_bboxes)               # 计算 预测框与边界框的giou,这里是指匹配上的边界框,维度是[N]
            giou_loss = 1.0 - giou                                         # giou loss 阈值范围是 [0, 2)
            self.loss_box_regression += giou_loss.mean()                   # 当前预测层的 giou_loss加入总体

            # 在yolov5中,使用box回归时计算的giou大小作为概率值, 但是, giou的值阈是(-1, 1], 因此应该对giou的值进行裁剪.
            objectness_ground_truth[gt_img_id, select_anchor_index, grid_y, grid_x] = giou.detach().clamp(0)    # 生成gt_objectness, 维度是[N, num_anchor, H, W]


            # 计算类别 loss
            # 如果类别数为1的话, 就直接使用objectness作为你的目标概率, 所以不需要重复计算目标概率了
            if self.num_classes > 1:
                # object_predict的维度是 [N, 5+num_classes], N表示匹配的目标数目
                # [cx, cy, w, h, objectness, class_1, class_2, ...]
                # 计算分类损失, 只在存在目标的位置计算
                pt_object_classification = object_predict[..., 5:]                      # 当前层预测的分类概率, 切记这是选出来的匹配上目标的位置
                gt_object_classification = torch.zeros_like(pt_object_classification)   # 维度是 [N, 20], N表示目标数, 20表示类别数
                
                # yolov5对object_loss和classification_loss均采用二元交叉熵损失
                # 区别是 一个是二元交叉熵, 一个是多个二元交叉熵
                # 在目标检测领域, 目前多数采用多个二元交叉熵代替多元交叉熵
                # 通常,分类领域采用多元交叉熵损失, 但是多标签分类(比如汽车、颜色)必须采用多个二元交叉熵
                # 二元交叉熵与多元交叉熵的区别
                # 二元交叉熵(Binary Cross Entropy): -sum(y*log(1-p) + (1-y)*log(p))      # 这里的p是经过sigmoid的
                # 多元交叉熵: -sum(y * log(p))                                           # 这里的p是经过softmax, 所有维度p的概率求和为1
                # 同样是多分类问题(区分: 猫、狗、猪)
                # 多个二元交叉熵
                #   -> 如果标签是猫
                #       预测结果是[为猫的概率, 为狗的概率, 为猪的概率], 概率都是[0, 1]
                #   -> 如果标签是猫、狗
                #       预测结果是[为猫的概率, 为狗的概率, 为猪的概率], 概率都是[0, 1]
                # 多元交叉熵
                #   -> 如果标签是猫
                #       预测结果是[为猫的概率, 为狗的概率, 为猪的概率], 概率都是[0, 1], 但是三者概率之和为1
                #   -> 如果标签是猫、狗
                #       不能采用这种方法
                # 在分类时, 可以对标签进行平滑,被称之为 LabelSmooth
                #   -> 方法1：
                #       labelsmooth1 = 如果e = 0.3，类别数为3的时候：  0.15, 0.7, 0.15
                #   -> 方法2：
                #       labelsmooth2 = 如果e = 0.3，类别数为3的时候：  0.1, 0.8, 0.1

                # 根据匹配成功的targets,以及它所对应的class_id构造label
                # gt_class_id 的维度是 [N], N 表示匹配的目标数
                # 这里最开始写错了，找了很久的bug，损失确实在降低，一直没有mAP结果
                # 这里错误的将torch.arrange(matched_extend_num_target),写成了matched_extend_num_target:，使得回归的类别概率都为零
                # 这个错误是极其致命且隐蔽的，引以为戒
                matched_extend_num_target = len(select_targets)
                gt_object_classification[torch.arange(matched_extend_num_target), gt_class_id] = 1.0           # 将每一个target对应的类别概率位置设置为1
                self.loss_classification += self.BCE_classification(pt_object_classification, gt_object_classification)

        # 无论是否匹配上目标都需要计算, 都需要计算objectness损失
        # 当没有匹配上目标时, 就将预测到的objectness回归到零
        self.loss_objectness += self.BCE_objectness(feature_map_objectness, objectness_ground_truth) * self.balance[ilevel]
        pass
    

    # 根据匹配策略, 构建标签
    def build_targets(self, targets, ilevel, feature_map_w, feature_map_h):
        """
            ### 函数说明
                 - 确定当前预测层能匹配上的targets以及与其对应的anchors
                 - targets: 当前批次图像所对应的所有目标
                 - ilevel: 当前是第几个预测层, 一共3个预测层 P8, P16, P32
                 - feature_map_w: 当前预测层对应的特征图的宽度
                 - feature_map_h: 当前预测层对应的特征图的高度
            ### 返回值
                - select_targets:      当前预测层可以匹配上anchor的targets, 包括扩充的正样本
                - select_anchor_index: 匹配上的targets对应的anchor的索引, 包括扩充的正样本
                - xy_offset:           目标中心的偏移量, 原始匹配的targets没有偏移量,扩充的正样本的偏移量, 主要用来计算扩充正样本的grid_x, grid_y
        """
        xy_offset = None
        feature_size_gain = targets.new_tensor([1, 1, feature_map_w, feature_map_h, feature_map_w, feature_map_h])
        targets_feature_scale = targets * feature_size_gain                                             # 把targets给弄到基于featuremap大小的框

        # 取出来当前预测层的anchors
        anchors = self.anchors[ilevel]                                  # 当前特征图对应的anchor 维度是 [3, 2]
        num_anchor = anchors.size(0)                                    # 当前特征图包括3个尺度的anchor

        # 计算宽宽比、高高比最大值, 用于为每个目标匹配当前层的anchor, 这里利用了广播机制
        anchors_wh = anchors.view(num_anchor, 1, 2)                                                      # 当前特征图尺度下的anchor宽高
        targets_feature_scale_wh = targets_feature_scale[:, [4, 5]].view(1, self.num_target, 2)          # 当前特征图尺度下的targets的宽高
        
        anchor_target_wh_ratio = anchors_wh / targets_feature_scale_wh                              # 计算anchor与target的宽高比, 维度是[3, M, 2]

        max_wh_ratio_value, _ = torch.max(anchor_target_wh_ratio, 1 / anchor_target_wh_ratio).max(dim=2)        # 维度是[3, M], 每个anchor与target的比例的最大值

        select_targets_mask = max_wh_ratio_value < self.anchor_t                                                # 维度是[3, M]
        select_targets = targets_feature_scale.repeat(num_anchor, 1, 1)[select_targets_mask]                    # 维度是[20, 6], 20是匹配上的targets
        select_anchor_index = torch.arange(num_anchor, device=self.device).view(num_anchor, 1).repeat(1, self.num_target)[select_targets_mask]        # 与选出targets对应的anchor, 维度是[20, 1]
        
        self.matched_num_per_target += torch.sum(select_targets_mask.detach().float(), dim=0)               # 记录每个target匹配的次数
        # print(self.matched_num_per_target)
        

        # 到此为止, 我们已经选出了当前预测层匹配成功的目标数以及其对应的anchor的索引
        # 接下来, 我们将会扩充正样本数量, yolov5；理论上扩充了2倍
        if len(select_targets) > 0:                                                         # 如果当前层匹配上了目标, 那就对目标扩充2倍
            select_targets_xy = select_targets[:, [2, 3]]                                   # 选出每个target的cx,cy, 维度是 [20, 2]
            xy_divided_one_remainder = select_targets_xy % 1.0                              # 计算cx,cy的小数部分, 确定在那个位置扩充, 维度是[20, 2]

            # 我们通常以方框左上角表示grid_x, grid_y
            # 定义中心位置, 以及上下边界
            coord_cell_center = 0.5                                                         # 表示一个像素的中心点, 根据判断在中心那边增加样本
            feature_map_low_boundary = 1.0                                                  # 当cx > 1, 左边才可以扩充, cy > 1, 上边才可以扩充
            feature_map_high_boundary = feature_size_gain[[2, 3]] - 1.0                     # 当cx < w-1, 右边才可以扩充, cy < h-1, 下面才可以扩充

            # 表示cx,cy的小数部分小于0.5的选出来,并且cx, cy两者必须大于下边界1, 这样才能扩充左边和上边
            # 比如 cx=7.2 可以扩充左边, cy = 7.4 可以扩充上边, 但是cx=0.3,是不可以扩充左边的,因为越界了
            # 这里选出来(值为TRUE)的都是可以扩充的目标
            # 这里的T表示转置, 很方便用于解包
            # 左上扩充
            expand_left_mask, expand_top_mask = ((xy_divided_one_remainder < coord_cell_center) & (select_targets_xy > feature_map_low_boundary)).T

            # 当cx, cy的小数部分大于0.5时, 我们需要在右边和下面两个方向扩充
            # 比如 cx=7.6 可以扩充右边, cy = 7.9 可以扩充下边, 但是cx=w-1,是不可以扩充右边的,因为越界了
            # 这里选出来的也就是可以在右边和下边扩充的目标
            # 右下扩充
            expand_right_mask, expand_bottom_mask = ((xy_divided_one_remainder > (1 - coord_cell_center)) & (select_targets_xy < feature_map_high_boundary)).T

            # 当cx=0.5或者cy=0.5时不进行扩充, 这种可能性很低
            # 理论上,这里扩充的正样本是2倍的目标数, 但是边界上不会扩充, 因此特殊情况下, 会小于2倍
            select_anchor_index = torch.cat([
                select_anchor_index,                        # 原始的目标匹配anchor索引, 维度是 [20]
                select_anchor_index[expand_left_mask],      # 左边 扩充目标对应的anchor索引, 维度是 [k]
                select_anchor_index[expand_top_mask],       # 上边 扩充目标对应的anchor索引, 维度是 [m]
                select_anchor_index[expand_right_mask],     # 右边 扩充目标对应的anchor索引, 维度是 [l]
                select_anchor_index[expand_bottom_mask]     # 下边 扩充目标对应的anchor索引, 维度是 [n]
            ], dim=0)                                       # 一般地, k+m+l+n=40, 维度是[60]
            select_targets = torch.cat([
                select_targets,                             # 原始的目标, 维度是 [20, 6]
                select_targets[expand_left_mask],           # 左边 扩充目标, 维度是 [k, 6]
                select_targets[expand_top_mask],            # 上边 扩充目, 维度是 [m, 6]
                select_targets[expand_right_mask],          # 右边 扩充目标, 维度是 [l, 6]
                select_targets[expand_bottom_mask]          # 下边 扩充目标, 维度是 [n, 6]
            ], dim=0)                                       # 一般地, k+m+l+n=40, 维度是[60, 6]

            # 至此, 正样本扩充完毕, 大致扩充了2倍

            # 建立offset, 使增加出来的样本的中心更够计算正确的 grid_x, grid_y
            # 这里为什么会有xy_offset呢？
            # 当我们扩充样本后,我们并不知道上面增加的样本对应的 grid_x, grid_y
            # 假设我们cx=7.499, 我们会在左边增加一个正样本, 那么如何计算它的 grid_x呢？
            # 我们这样计算 (cx-1.0).long() = 6, 它回归的具体偏移量就是 1.499
            # 注意: 所有类型转换操作, 都是取其整数位, torch.long()既不是向上取整, 也不是向下取整, 直接去除小数部分
            # 下面的xy_offset就是在计算目标属于那个网格预测时候使用的
            # grid_xy = (select_targets[:, [2, 3]] - xy_offset).long()
            # 这样就确定了那个目标由那个网格参数预测
            xy_offset = torch.zeros_like(select_targets_xy)                   # 原始匹配的目标不需要偏移量,就可以计算正确的grid_x, grid_y.维度是[20，2]           
            xy_offset = torch.cat([
                xy_offset,                                                    # 原始匹配上的目标不需要添加偏移量就可以正确计算 grid_x, grid_y
                xy_offset[expand_left_mask] + self.offset_boundary[0],        # 左边扩充, 计算其grid_x, grid_y
                xy_offset[expand_top_mask] + self.offset_boundary[1],         # 上边扩充, 计算其grid_x, grid_y
                xy_offset[expand_right_mask] + self.offset_boundary[2],       # 有边扩充, 计算其grid_x, grid_y
                xy_offset[expand_bottom_mask] + self.offset_boundary[3]       # 下边扩充, 计算其grid_x, grid_y
            ], dim=0) * coord_cell_center                                     # 我觉得这里没必要乘, 最终计算的 grid_x, grid_y是一样的, 因为long是干掉小数

            
        return select_targets, select_anchor_index, xy_offset







if __name__ == "__main__":
    # hyp = {"anchor_t": 4.0, "box": 0.05, "cls": 0.01, "obj": 0.02}
    hyp = {"anchor_t": 4.0, "box": 0.05, "cls": 0.01, "obj": 0.02, "label_smoothing": 0.15}
    device = "cuda"
    anchors = [
        [10,13, 16,30, 33,23],  # P3/8
        [30,61, 62,45, 59,119],  # P4/16
        [116,90, 156,198, 373,326]  # P5/32
        ]
    loss = ComputeLoss(20, anchors, hyp)
    predicts_0 = torch.ones((8, 75, 80, 80), dtype=torch.float32).to(device)
    predicts_1 = torch.ones((8, 75, 40, 40), dtype=torch.float32).to(device)
    predicts_2 = torch.ones((8, 75, 20, 20), dtype=torch.float32).to(device)
    predicts = [predicts_0, predicts_1, predicts_2]
    targets = torch.ones((10, 6), dtype=torch.float32).to(device)
    targets[:, 2:]  = targets[:, 2: ] * 0.09
    
    total_loss, part_loss, [matched_num_mean, unmatched_num_target] = loss(predicts, targets)

    print(total_loss, part_loss)
    # print(sum(part_loss)*8)
    
    # print(matched_num_mean)
    # print(unmatched_num_target)











































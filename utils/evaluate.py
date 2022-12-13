#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2022/10/15 15:00:58
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   mAP 计算
'''


import os
import cv2
import sys
import numpy as np
import torch

from pathlib import Path


from tqdm import tqdm
import torchvision

sys.path.append("..")


from utils.dataset import LoadImagesAndLabels
from torch.utils.data.dataloader import DataLoader


class MAPTool(object):
    """
        ### 函数说明
            - 计算mAP
            - 分为三种计算模式, "interp11", "interp101", "continuous"
            - 直接调用COCO  API
    """
    def __init__(self, all_pt_bboxes_dict, all_gt_bboxes_dict, num_classes=20, method="interp11", max_det=100):
        self.method = method
        assert self.method in ["interp11", "interp101", "continuous"], f'Parameter is error, ["interp11", "interp101", "continuous"]'
        self.num_classes = num_classes
        self.max_det = max_det
        self.all_pt_bboxes_dict = all_pt_bboxes_dict
        self.all_gt_bboxes_dict = all_gt_bboxes_dict
        pass

    def ap_classes(self, ap_iou_thres):
        ap_list_all_class = np.zeros((self.num_classes,))
        for cls_id in range(self.num_classes):
            ap = self.ap_per_class(self.all_pt_bboxes_dict, self.all_gt_bboxes_dict, cls_id, ap_iou_thres)
            ap_list_all_class[cls_id] = ap
        return ap_list_all_class


    def mAP(self, ap_iou_thres):
        """
            ### 函数说明
                - 所谓的mAP是指,多个类别的AP的均值,就是mean
                - 先计算每个类别的AP,然后再去均值就可以了
        """
        ap_list_all_class = self.ap_classes(ap_iou_thres)
        mAP_value = np.mean(ap_list_all_class)
        return mAP_value
    

    def ap_per_class(self, all_pt_bboxes_dict, all_gt_bboxes_dict, class_id=0, ap_iou_thres=0.5):
        """
            ### 函数说明
                - 针对每一个图像,选择该图像上指定的类别(compute_classes_index)的groundtruth和detection框
                - 构建matched_table,格式如下:
                    - 每一行的格式是：[confidence, matched_iou, matched_groundtruth_index, image_id]
                    - 行数是所有图像的指定类别(compute_classes_index)的detection数量总和
                    - matched_groundtruth_index是当前detection的框,与其图像上所有同类别groundtruth框IoU最大的那个的索引
                    - matched_iou则是与当前detection最匹配的matched_groundtruth_index的IoU
                - 对matched_table进行排序,基于confidence倒序(大的在前面，小的在后面）
                - 同时统计该类别的groundtruth的总数量备用
        """
        matched_table = []                                                 # 所有图像，类别为零的检测框与groundtruth为零的标注边界框的匹配情况
        sum_groundtruth = 0                                                # 所有图像, 类别0的标注边界框的个数

        # print("检测到的目标数:", len(all_pt_bboxes_dict))
        for img_id in all_pt_bboxes_dict:
            # print(all_pt_bboxes_dict[img_id].shape)
            select_pt_boxes = np.array(list(filter(lambda x: x[-1] == class_id, all_pt_bboxes_dict[img_id])))              # 将预测的类别为0的边界框取出来
            select_gt_boxes = np.array(list(filter(lambda x: x[-1] == class_id, all_gt_bboxes_dict[img_id])))
            # pt_mask = all_pt_bboxes_dict[img_id][:, -1] == class_id        # 选出当前类别的预测框
            # gt_mask = all_gt_bboxes_dict[img_id][:, -1] == class_id        # 选出当前类别的 gt框
            # select_pt_boxes = all_pt_bboxes_dict[img_id][pt_mask]       
            # select_gt_boxes = all_gt_bboxes_dict[img_id][gt_mask]

            cur_num_pt = len(select_pt_boxes)                              # 这个图像上，类别为0，检测到的对象数目
            cur_num_gt = len(select_gt_boxes)                              # 当前这张图像上，类别为0，实际的目标数目
            num_use_pt = min(cur_num_pt, self.max_det)                     # 只对前100个预测框计算
            sum_groundtruth += cur_num_gt                                  # 记录所有正阳的数目,为了计算召回率
            
            if num_use_pt == 0:                                            # 当检测为空时,跳到下一张图像的预测结果,筛选 0 类别
                continue
            if cur_num_gt == 0:
                for i in range(num_use_pt):
                    confidence = select_pt_boxes[i, 4]
                    matched_table.append([confidence, -1, -1, img_id])
                continue

            
            sdt_bboxes = select_pt_boxes[:num_use_pt, :4]                                           # 只选出前100个结果比对
            sgt_bboxes = select_gt_boxes[..., :4]
            # shape=num_groundtruth x num_detection, 每一列表示预测框与每一个标签框的IOU
            gt_pt_iou = self.bboxes_iou(sgt_bboxes, sdt_bboxes)
            for idx in range(num_use_pt):                                 # 遍历每一个检测框,构造相应的matched_table输出
                confidence = select_pt_boxes[idx, 4]                      # 置信度得分,这里是 objectness * classification
                matched_gt_idx = gt_pt_iou[:, idx].argmax()               # 找出最大的IOU的索引,就是当前检测框匹配的gt框
                matched_gt_iou = gt_pt_iou[matched_gt_idx, idx]           # gt框与当前检测框最大的IOU的值
                matched_table.append([confidence, matched_gt_iou, matched_gt_idx, img_id])
                
        
        if sum_groundtruth == 0:                                          # 如果没有当前类别的gt标签,表示预测的全是错的
            return 0
        if len(matched_table) == 0:                                       # 当没匹配上任何目标时,表示模型效果很差
            return 0
        
        # 按照confidence对所有的匹配框排序, 开始计算P-R曲线
        matched_table = sorted(matched_table, key= lambda x: x[0], reverse=True)    

        # 根据ap阈值计算P-R曲线
        precision, recall = self.calculate_PR(matched_table, sum_groundtruth, ap_iou_thres)
        
        

        # 计算P-R曲线下的面积
        ap = self.intergrate_area_under_curve(precision, recall, method=self.method)

        return ap

    
    def intergrate_area_under_curve(self, precision, recall, method="interp11"):
        """
            ### 函数说明
                - Source: https://github.com/rbgirshick/py-faster-rcnn.
                - 用不同的方法计算ap的值
                - 方法包括: interp11(voc2007), interp101(COCO), continuous(voc2012)
        """
        # 平滑P-R曲线,用于计算ap
        mpre, mrec = self.smooth_PR(precision, recall)

        if method == "interp101":                               # COCO评价方法
            x = np.linspace(0, 1, 101)                          # 获取插值点
            ap = np.mean(np.interp(x, mrec, mpre))              # 积分, 直接取均值,论文中都是这么做的
        elif method == "interp11":                              # VOC2007评价方法
            x = np.linspace(0, 1, 11)                           # 获取插值点
            ap = np.mean(np.interp(x, mrec, mpre))              # 积分, 直接取均值,论文中都是这么做的
        else:                                                   # continuous, VOC2012的评价方法
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes (VOC2012)
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
        return ap


    def smooth_PR(self, precision, recall):
        """
            ### 函数说明
                - 产生的P-R曲线会存在凹进去的锯齿,需要进行平滑处理(也就是包络线)
                - 使用的重要函数是np.maximum.accumulate(x), np.flip()
                - 包络线的计算方法:
                    - precision的值从大到小,所以计算方法是从右往左,用右边的最大值替代前面的小于该值的值
                - np.maximum.accumulate(x)是从左到右的,因此计算precision时候需要反转
                - 需要在precision与recall两端插值,便于后续计算插值点
        """
        
        
        
        mpre = np.concatenate(([0.], precision, [0.]))                                    # 因为它是纵坐标,两端添加零
        mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1e-3, 1.)]))               # 它是横坐标,右端往后移动一丢丢

        # 计算precision的包络线
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        
        return mpre, mrec


    def calculate_PR(self, matched_table, num_gts, ap_iou_thres=0.5):
        """
            ### 函数说明
                - matched_table是所有验证集上类别为0的检测框,用于计算该类别的P-R曲线
                - num_gts是所有验证集上类别为0的标注框, 计算recall
                - ap_iou_thres是计算AP时的阈值, 比如@0.5
            ### 返回值
                - P-R曲线,也就是precision和recall的值
        """
        num_dets = len(matched_table)                                       # 检测框的数目,用于计算precision=TP/num_dets
        true_positive = np.zeros((num_dets,))                               # TP 真阳性
        gt_bboxes_seen_map = {item[-1]: set() for item in matched_table}    # 用img_id作为键, 键值用一个set,存储被seen过的gt框的id
        for index, (confidence, matched_iou, matched_gt_idx, img_id) in enumerate(matched_table):
            seen_gt_idx_set = gt_bboxes_seen_map[img_id]                    # 获取当前图像gt标签的匹配情况
            if matched_iou >= ap_iou_thres:                                 # 如果当前检测框与gt标签的IOU大于阈值
                if matched_gt_idx not in seen_gt_idx_set:                   # 并且gt_idx之前未被匹配过
                    true_positive[index] = 1                                # 那么当前检测框被设置为TP,表示匹配成功
                    seen_gt_idx_set.add(matched_gt_idx)                     # 并将当前的gt_idx设置为被看过了,也就是有检测框匹配上了

        num_predicts = np.ones((num_dets,))
        accumulate_num_predicts = np.cumsum(num_predicts)                   # 累加检测到的边界框数目,为了计算precision,作为分母
        accumulate_true_positive = np.cumsum(true_positive)                 # 累加预测正确的边界框
        precision = accumulate_true_positive / accumulate_num_predicts      # 预测出来的框中匹配上gt框的比例,精确率
        recall = accumulate_true_positive / num_gts                          # 召回率
        
        return precision, recall

    
    def bboxes_iou(self, gt_boxes, dt_boxes):
        '''
            # 计算bboxes之间的IOU,计算所有检测框与所有标注框之间的IOU值
            # 重点是需要用到numpy广播机制
            # IoU的计算,此时是实现可以允许a是jx,b是1xk。结果是jxk
            gt_boxes: type=ndarray, shape=[M*4]
            dt_boxes: type=ndarray, shape=[N*4]
        '''
        gt_boxes = gt_boxes.T.reshape(4, -1, 1)                                                     # 将形状变为 4*M*1， 添加一个维度1，是为了应用广播机制
        dt_boxes = dt_boxes.T.reshape(4, 1, -1)                                                     # 将形状变为 4*1*N

        gt_left, gt_top, gt_right, gt_bottom = [gt_boxes[i] for i in range(4)]
        dt_left, dt_top, dt_right, dt_bottom = [dt_boxes[i] for i in range(4)]

        gt_width, gt_height = (gt_right - gt_left + 1), (gt_bottom - gt_top + 1)                        # 标注边界框的宽度和高度
        dt_width, dt_height = dt_right - dt_left + 1, dt_bottom - dt_top + 1                        # 预测边界框的宽度和高度

        cross_left   = np.maximum(gt_left, dt_left)
        cross_top    = np.maximum(gt_top, dt_top)
        cross_right  = np.minimum(gt_right, dt_right)
        cross_bottom = np.minimum(gt_bottom, dt_bottom)

        cross_area = (cross_right - cross_left + 1).clip(0) * (cross_bottom - cross_top + 1).clip(0)        # 交集的面积
        union_area = gt_width * gt_height + dt_width * dt_height - cross_area
        
        return cross_area / union_area




def xywh2xyrb(bboxes):
    """
        ### 函数说明
            - 转换多个边界框的坐标
            - bboxes: [N, 4], [x, y, w, h]
    """
    left   = bboxes[:, 0] - (bboxes[:, 2] - 1) * 0.5
    right  = bboxes[:, 0] + (bboxes[:, 2] - 1) * 0.5
    top    = bboxes[:, 1] - (bboxes[:, 3] - 1) * 0.5
    bottom = bboxes[:, 1] + (bboxes[:, 3] - 1) * 0.5
    bboxes[:, 0] = left
    bboxes[:, 1] = top
    bboxes[:, 2] = right
    bboxes[:, 3] = bottom
    return bboxes



# batched NMS, 对多张图像的推理结果,执行非极大值抑制,得到预测框
def non_max_suppression(predictions, iou_thres=0.5, conf_thres=0.001, max_output_det=30000):
    """
        ### 参数
            - predictions: [[N, 5+20], [N, 5+20], ...], [x, y, w, h, conf, cls_p1, cls_p2...]
            - iou_thres: 计算mAP时, 应该设置为0.5
            - conf_thres: 计算mAP时, 应该设置为0.001
        ### 执行过程说明
            - 根据conf_thresh筛选预测框, 计算mAP会设置为0.001, 在detect时, 会设置的大一些,比如0.2-0.5
            - yolov5采用conf_thresh筛选2次, 一次是objectnes > conf_thres, 一次是 confidence_score > conf_thres
            - yolov5中提到的confidence_score是指 objectness * classification
            - 我觉得筛选两次和一次没什么区别,主要保证 score > conf_thres
        ### 返回值
            - output是一个list
            - list中的每一个元素就是一个图像的检测框
            - 检测框的返回结构是[left, top, right, bottom, conf_score, class_id]
    """
    # 检查输入参数
    assert isinstance(predictions, torch.Tensor), f'Predictions Tensor dims must: [bs, num_box, 5+cls]'
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are [0, 1].'
    assert 0 <= iou_thres <= 1, f'Invalid IoU threshold {iou_thres}, valid values are [0, 1].'

    max_wh = 7680                                                                # box 宽度和高度的最大值
    max_nms = 30000                                                              # torchvision.ops.nms()处理的最大box数量

    bs = predictions.shape[0]                                                    # 批量大小
    output = [torch.zeros((0, 6), device=predictions.device)] * bs               # 输出列表,每个元素就是一张图像的预测结果

    # 初步筛选,这一步可以不做,这里主要减少候选框数量,可提速
    # objectness > conf_thres
    select_mask = predictions[..., 4] > conf_thres                  # 维度是[bs, num_boxes]=[bs, 25200]
    
    for i, predict in enumerate(predictions):                       # 遍历每一张图像的预测结果
        predict = predict[select_mask[i]]                           # 选出 objectness > conf_thres部分检测结果, 维度是 [N, 5+cls]
        bbox = xywh2xyrb(predict[:, :4])                            # 转换边界框, 从 xywh->left,top,right,bottom, clamp(0, 640-1)?
        predict[:, 5:] *= predict[:, 4:5]                           # 计算score = objectness * classification
        score, cls_id = predict[:, 5:].max(1, keepdim=True)
        predict = torch.cat((bbox, score, cls_id.float()), dim=1)[score.view(-1) > conf_thres]
        
        num_box = predict.shape[0]
        
        if not num_box:                                 # 没有检测框,直接跳过
            continue
        elif num_box > max_nms:                         # 超过torchvision.ops.nms()处理的最大box数量, 排序之后取前面部分
            predict = predict[predict[:, 4].argsort(descending=True)[:max_nms]]
        else:
            predict = predict[predict[:, 4].argsort(descending=True)]

        bboxes, scores, cls_ids = predict[:, :4], predict[:, 4], predict[:, 5]
        keep_idx = torchvision.ops.batched_nms(bboxes, scores, cls_ids, iou_thres)
        if keep_idx.shape[0] > max_output_det:                                       #只输出 一张图像的最大预测数目的边界框,和计算mAP还不太一样
            keep_idx = keep_idx[:max_output_det]
        
        output[i] = predict[keep_idx]

    return output


def convert_gt_bboxes_format(gt_bboxes, img_size=640):
    """
        ### 函数说明
            - gt_bboxes: 维度是[N, 6], [img_id, class_id, x, y, w, h]
            - 需要将其转换为 [left, top, right, bottom, 0, class_id]
    """
    gt_bboxes = gt_bboxes[:, [2, 3, 4, 5, 0, 1]] * gt_bboxes.new_tensor([img_size, img_size, img_size, img_size, 1, 1])
    gt_bboxes[:, :4] = xywh2xyrb(gt_bboxes[:, :4])

    return gt_bboxes




# 保存预测结果到内存或者本地, 加载gt_bboxes到内存或者本地
# 模型推理得到预测框,然后进行筛选出符合条件的最终预测框, conf > 阈值, 并且进行类间的NMS
# 加载gt框到内存
def generate_predicts_targets_dict(model, val_txt_path, img_size=640, batch_size=16, num_workers=16,
                                   max_det=30000, nms_thres=0.5, conf_thres=0.001, device="cuda"):
    """
        ### 函数说明
            - 根据训练的模型对验证集数据进行推理,并将结果保存,用于计算mAP
            - 推理的结果应该先通过 conf_thres筛选,然后再进行类内的NMS. 最后将结果变换为想要的格式[left,top,right,bottom,conf_score,cls_id]
            - 最后,将每个图像的名称作为键与其检测框匹配,并将所有的图像与其检测结果存储到dict中,并保存起来(非必须)
    """
    val_datasets = LoadImagesAndLabels(val_txt_path, img_size, augment=False, cache_images=True, prefix="val")
    val_dataloader = DataLoader(val_datasets, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=num_workers, 
                                collate_fn=val_datasets.collate_fn,
                                pin_memory=True)
    model = model.to(device).eval()
    
    all_pt_bboxes = dict()
    all_gt_bboxes = dict()
    
    # # 检查本地是否缓存了制作好的target信息，是处理成计算mAP要求的格式
    # flag= False
    # all_gt_bboxes_file = os.path.join(Path(val_txt_path).parent, "val_all_gt_bboxes.npy")
    # if os.path.exists(all_gt_bboxes_file):
    #     flag = True
    #     all_gt_bboxes = np.load(all_gt_bboxes_file, allow_pickle=True).item()
    # else:
    #     # print(all_gt_bboxes_file, " not exist!")
    #     pass
    
    # 从验证集随机选择一张图像
    # 用于将真实边界框和预测框展示到visdom去
    rand_i = np.random.randint(1, len(val_datasets))
    show_gt_pt_dict = dict()
    
    
    with torch.no_grad():
        pbar = tqdm(enumerate(val_dataloader), total=len(val_datasets)//batch_size, desc="Compute mAP...")
        for i, (images, labels, visual_info) in pbar:
            tmp_img = images.numpy()                                                                # 用于模型检测效果展示
            images, labels = images.to(device).float() / 255, labels.to(device)
            predicts = model(images).detach()
            predicts = non_max_suppression(predicts, nms_thres, conf_thres, max_det)        # 对预测结果进行batched_nms,                       
            for j in range(len(predicts)):
                index = i * batch_size + j                                                  # 计算计算到了第几个图像用于将图像名称与预测框对应    
                img_id = val_datasets.image_files[index]                                    # 图像ID            
                pt_bboxes = predicts[j]                                                     # 维度是[M, 6], [left, top, right, bottom, score, clas_id]
                
                gt_bboxes = labels[labels[:, 0] == j].detach()                          # 维度是 [K, 6], [img_id, class_id, x, y, w, h]
                gt_bboxes = convert_gt_bboxes_format(gt_bboxes)                         # 维度是 [K, 6], [left, top, right, bottom, 0, clas_id]
                
                all_gt_bboxes[img_id] = gt_bboxes.cpu().numpy()                         
                all_pt_bboxes[img_id] = pt_bboxes.cpu().numpy()
                
                if index == rand_i:
                    show_gt_pt_dict["image"] = tmp_img[j]
                    show_gt_pt_dict["gt_bboxes"] = gt_bboxes
                    show_gt_pt_dict["pt_bboxes"] = pt_bboxes

        # np.save(all_gt_bboxes_file, all_gt_bboxes)                                          # 将标签信息保存到本地    
        
        # 这里的all_pt_bboxes, all_gt_bboxes标签,都是在img_size=640尺度下的边界框 
        # show_gt_pt_dict用于观察当前模型的检测效果

        return all_pt_bboxes, all_gt_bboxes, show_gt_pt_dict


def recover_bbox(img_size, ori_h, ori_w, pt_bboxes):
    """
        ### 函数说明
            1. img_size, 表示推理时的图像大小
            2. ori_h, 表示原始图像的高度
            3. ori_w, 表示原始图像的宽度
            4. pt_bboxes, 表示在img_size尺度下的检测框, 维度是[M, 6], [left, top, right, bottom, score, clas_id]
    """
    pt_bboxes = pt_bboxes.copy()
    scale = min(img_size / ori_w, img_size / ori_h)
    offset_x, offset_y = (img_size/2 - (ori_w*scale)/2), (img_size/2 - (ori_h*scale)/2)
    pt_bboxes[:, :4] -= np.array([offset_x, offset_y, offset_x, offset_y])
    pt_bboxes[:, :4] *= np.array([1/scale, 1/scale, 1/scale, 1/scale])
    
    # pt_bboxes[:, 0].clamp_(0, ori_w)  # x1
    # pt_bboxes[:, 1].clamp_(0, ori_h)  # y1
    # pt_bboxes[:, 2].clamp_(0, ori_w)  # x2
    # pt_bboxes[:, 3].clamp_(0, ori_h)  # y2
    
    return pt_bboxes



# 根据训练的模型，评估数据集，计算mAP
def estimate(model, val_data_path, method="interp101", num_cls=20, image_size=640, batch_size=16, num_workers=16,
                                   nms_max_output_det=30000, map_max_det=100, nms_thres=0.5, conf_thres=0.001, device="cuda"):
    """_summary_

    参数说明:
        model (_type_): 网络模型
        val_data_path (_type_): 验证集文件路径
        method (str, optional): mAP的计算方法. Defaults to "interp101".
        num_cls (int, optional): 类别数目. Defaults to 20.
        image_size (int, optional): 评估验证模型时输入分辨率. Defaults to 640.
        batch_size (int, optional): 评估推理时批的大小. Defaults to 16.
        num_workers (int, optional): 线程数目. Defaults to 16.
        nms_max_output_det (int, optional): NMS后保留的最大框数目. Defaults to 30000.
        map_max_det (int, optional): 计算mAP时通常选取前100个置信度最大的框(在每张图像上). Defaults to 100.
        nms_thres (float, optional): NMS的阈值. Defaults to 0.5.
        conf_thres (float, optional): 计算mAP时通常选取较小的置信度阈值,确保较高的召回率. Defaults to 0.001.
        device (str, optional): 评估时,选择推理设备. Defaults to "cuda".
    """
    # print("Starting evalate validation dataset...")
    model.eval()
    
    # all_pt_bboxes_dict = np.load("../data/all_pt_bboxes.npy", allow_pickle=True).item()
    # all_gt_bboxes_dict = np.load("../data/all_gt_bboxes.npy", allow_pickle=True).item()
    all_pt_bboxes_dict, all_gt_bboxes_dict, show_gt_pt_dict = generate_predicts_targets_dict(model, val_data_path)
    

    image = show_gt_pt_dict["image"]
    image = np.ascontiguousarray(image.transpose(1, 2, 0)[..., ::-1])
    for x, y, r, b, _, _ in show_gt_pt_dict["gt_bboxes"]:
        x, y, r, b = int(x), int(y), int(r), int(b)
        cv2.rectangle(image, (x, y), (r, b), color=(0, 0, 255), thickness=2)
    n = 0
    for x, y, r, b, _, _ in show_gt_pt_dict["pt_bboxes"]:
        n += 1
        if n < 20:
            x, y, r, b = int(x), int(y), int(r), int(b)
            cv2.rectangle(image, (x, y), (r, b), color=(255, 0, 0), thickness=2)
        
    
    num_pt = 0
    num_gt = 0
    for key in all_pt_bboxes_dict:
        num_pt += len(all_pt_bboxes_dict[key])
        num_gt += len(all_gt_bboxes_dict[key])
    
    print("num_pt: ", num_pt)
    print("num_gt: ", num_gt)
    map_tool = MAPTool(all_pt_bboxes_dict, all_gt_bboxes_dict, num_classes=num_cls, method="interp101", max_det=map_max_det)
    
    mAP_list = []

    for iou_thres in range(50, 100, 5):
        map_val = map_tool.mAP(iou_thres/100.0)
        mAP_list.append(map_val)
        
    # print(mAP_list)
    # print("mAP@0.50     = {:.3f}".format(mAP_list[0]))
    # print("mAP@0.75     = {:.3f}".format(mAP_list[5]))
    # print("mAP@0.5:0.95 = {:.3f}".format(np.mean(np.array(mAP_list))))
    
    
    return mAP_list[0], mAP_list[5], np.mean(np.array(mAP_list)), image
    
    
    



if __name__ == "__main__":
    from dataset import LoadImagesAndLabels
    from torch.utils.data.dataloader import DataLoader
    import cv2
    import sys
    
    from utils.general import setup_random_seed
    from models.yolo import YoloV5

    val_txt_path = "/workspace/datasets/PASCAL_VOC2007/VOC2007_trainval/debug.txt"
    # setup_random_seed(3)
    model = YoloV5(yaml_cfg_file="/workspace/yolov5-pro/models/yolov5s-v2.yaml")
    # generate_predicts_targets_dict(model, val_txt_path)

    # all_pt_bboxes_dict = np.load("../data/all_pt_bboxes.npy", allow_pickle=True).item()
    # all_gt_bboxes_dict = np.load("../data/all_gt_bboxes.npy", allow_pickle=True).item()

   

    # map_tool = MAPTool(all_pt_bboxes_dict, all_gt_bboxes_dict, num_classes=20, method="interp101")
    # map = map_tool.mAP(0.75)

    # print(f'map@0.75={map:.3f}')
    
    AP50, AP75, mAP = estimate(model, val_txt_path)
    print(AP50, AP75, mAP)
    






















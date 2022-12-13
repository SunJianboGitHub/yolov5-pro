

import os
import cv2
import sys
import json
import torch
import torchvision

import numpy as np

from tqdm import tqdm
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("..")

from utils.dataset import create_dataloader


# 这个就是需要提供的label_map的样式
VOC_NAMES = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",  "sheep",  "sofa",  "train", "tvmonitor"]



# 利用COCO API 计算mAP
class COCOmAP:
    
    def __init__(self, gt_json_dir, pt_json_dir, label_map, prefix):

        assert os.listdir(gt_json_dir) == os.listdir(pt_json_dir), "gt_json_files != pt_json_files"
        assert isinstance(label_map, list), "label_map Type Error, must list type!"
        
        start_id = 10000000                                                                                 # 用于设置图像ID,因为COCO要求是整数
        self.num_gt = 0
        self.num_pt = 0
        
        # 制作gt_coco.json
        root_dir = Path(gt_json_dir).parent
        self.gt_coco_json_file = os.path.join(root_dir, prefix + "_gt_coco.json")
        if not os.path.exists(self.gt_coco_json_file):
            self.generate_gt_coco_json(gt_json_dir, label_map, prefix=prefix, start_img_id=start_id)
        else:
            with open(self.gt_coco_json_file, "r") as fr:
                gt_coco_dict = json.load(fr)
            self.num_gt = len(gt_coco_dict["annotations"])
            
        # 制作预测的json数据
        self.anno_dets = self.generate_pt_json_list(pt_json_dir, start_img_id=start_id)
        
        # 到此,就可以调用 COCO API 计算mAP
        
        pass
        
    
    # 根据预测的json文件制作coco格式的字典列表
    def generate_pt_json_list(self, pt_json_dir, start_img_id):
        
        anno_dets = []
        for json_file in os.listdir(pt_json_dir):
            json_file = os.path.join(pt_json_dir, json_file)
            pt_bboxes = self.read_json(json_file)
            
            for left, top, right, bottom, score, cls_id in pt_bboxes:
                width, height = right - left + 1, bottom - top + 1
                # 这里写成了[left, right, width, height]排查了一周错误
                # 简直吐血了,一定要仔细、仔细
                object_item = {"image_id": start_img_id, "category_id": int(cls_id), 
                               "score": score, "bbox":[left, top, width, height]}               
                anno_dets.append(object_item)
            
            start_img_id += 1
            
        self.num_pt = len(anno_dets)
        
        return anno_dets

        
        
    # 根据真实标签制作gt_coco.json文件
    def generate_gt_coco_json(self, gt_json_dir, label_map, prefix, start_img_id):
        images = []
        annotations = []
        categories = []
        anno_id = 0
        
        # 处理类别信息
        for label, label_name in enumerate(label_map):
            categories.append({"supercategory": label_name, 
                               "id": int(label), 
                               "name": label_name})                                     # 添加类别信息
        
        for json_file in os.listdir(gt_json_dir):                                       # 遍历所有的gt文件
            json_file = os.path.join(gt_json_dir, json_file)                            # 拼接完整的文件路径
            gt_bboxes = self.read_json(json_file)                                       # 读取标注文件中的边界框
            images.append({"id": start_img_id})                                         # 添加所有的图像ID
            
            for left, top, right, bottom, _, cls_id in gt_bboxes:                       # 遍历当前图像对应的边界框
                anno_id += 1                                                            # 创建边界框的唯一ID
                width, height = right - left + 1, bottom - top + 1                      # 计算边界框的宽度和高度
                annotations.append({
                    "image_id": start_img_id, "id": anno_id, "category_id": int(cls_id),
                    "bbox": [left, top, width, height], "iscrowd": 0, "area": width * height
                })                                                                      # 添加所有的边界框信息
                
            start_img_id += 1
            pass
        
        self.num_gt = len(annotations)
        
        gt_coco_dict = {"images": images, "annotations": annotations, "categories": categories}
        
        with open(self.gt_coco_json_file, "w") as fw:
            json.dump(gt_coco_dict, fw)
            
    
    
    
    # 读取一个json文件
    def read_json(self, json_file):
        with open(json_file, "r") as fr:
            return json.load(fr)
        
    

    def evaluate(self):
        
        # 表示未检测看到任何目标
        if len(self.anno_dets) == 0:
            print("Model can not detect object, pt_anno_coco_json = [], So do not evaluate...")
            return 0, 0, 0
        
        cocoGt = COCO(self.gt_coco_json_file)
        cocoDt = cocoGt.loadRes(self.anno_dets)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP, AP50, AP75 = cocoEval.stats[:3]
        return mAP, AP50, AP75





# 保存一个真实标签或者预测标签
def save_one_json(img_file, bbox_info):
    assert isinstance(bbox_info, np.ndarray), "bbox_info Type Error!"
    assert img_file.endswith(".json")
    if not os.path.exists(Path(img_file).parent):
        os.makedirs(Path(img_file).parent)
    with open(img_file, "w") as fw:
        json.dump(bbox_info.tolist(), fw, ensure_ascii=False)


# 转换坐标, 这里是像素坐标的转换, [x, y, w, h] -> [left, top, right, bottom]
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


# 先将归一化坐标转换为像素坐标,然后再转换为[x, y, w, h] -> [left, top, right, bottom]
def convert_gt_bboxes_format(gt_bboxes, img_size=640):
    """
        ### 函数说明
            - gt_bboxes: 维度是[N, 6], [img_id, class_id, x, y, w, h]
            - 需要将其转换为 [left, top, right, bottom, 0, class_id]
    """
    gt_bboxes = gt_bboxes[:, [2, 3, 4, 5, 0, 1]] * gt_bboxes.new_tensor([img_size, img_size, img_size, img_size, 1, 1])
    gt_bboxes[:, :4] = xywh2xyrb(gt_bboxes[:, :4])

    return gt_bboxes


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



# 将预测边界框进行非极大值抑制,得到最终预测,并将预测结果和标注一起显示
def show_compare_image(show_gt_pt_dict, label_map, conf_thres=0.25):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = show_gt_pt_dict["image"]
    gt_bboxes = show_gt_pt_dict["gt_bboxes"]
    pt_bboxes = show_gt_pt_dict["pt_bboxes"]
    
    image = np.ascontiguousarray(image.transpose(1, 2, 0)[..., ::-1])
    for x, y, r, b, _, cls_id in gt_bboxes:
        x, y, r, b = int(x), int(y), int(r), int(b)
        cv2.rectangle(image, (x, y), (r, b), color=(0, 0, 255), thickness=2)
        cv2.putText(image, label_map[int(cls_id)], (x, y), font, fontScale=0.5, color=(0, 0, 255), thickness=1)
        
    # 这里的pt_bboxes已经进行了NMS,只是置信度阈值较低,在这里采用置信度阈值重新帅选。
    # 阈值这里设置为 0.25
    for x, y, r, b, score, cls_id in pt_bboxes:
        if score >= conf_thres:
            x, y, r, b = int(x), int(y), int(r), int(b)
            cv2.rectangle(image, (x, y), (r, b), color=(255, 0, 0), thickness=2)
            cv2.putText(image, label_map[int(cls_id)], (x, y), font, fontScale=0.5, color=(255, 0, 0), thickness=1)
    return image



# 根据模型和数据集生成gt/00001.json和pt/00001.json
def generate_gt_pt_json(model, val_img_txt, prefix, img_size=640, batch_size=32, 
                       num_workers=16, max_det=30000, nms_thres=0.5, conf_thres=0.001, device="cuda"):
    data_loader, data_sets = create_dataloader(datas_path=val_img_txt,
                                               hyp=None,
                                               shuffle=False,
                                               augment=False,
                                               mixed_aug=False,
                                               cache_images=False,
                                               mosaic_nums=[4],
                                               prefix=prefix,
                                               batch_size=batch_size,
                                               img_size=img_size,
                                               num_workers=num_workers,
                                               border_fill_value=114)
    
    model.to(device).eval()
    
    save_dir = os.path.join(str(Path(val_img_txt).parent), "caches")
    save_gt_path = os.path.join(save_dir, prefix + "_groundtruths_json")
    save_pt_path = os.path.join(save_dir, prefix + "_predictions_json")
    
    # 从验证集随机选择一张图像
    # 用于将真实边界框和预测框展示到visdom去
    rand_i = np.random.randint(1, len(data_sets))
    show_gt_pt_dict = dict()
    
    with torch.no_grad():
        pbar = pbar = tqdm(enumerate(data_loader), total=len(data_sets)//batch_size, desc="Compute mAP...")
        for i, (images, labels, visual_info) in pbar:
            tmp_img = images.numpy()                                                                # 用于模型检测效果展示
            images = images.to(device, non_blocking=True).float() / 255
            predictions = model(images).detach()
            predictions = non_max_suppression(predictions, iou_thres=nms_thres, conf_thres=conf_thres, max_output_det=max_det)
            
            for j in range(len(predictions)):
                index = i * batch_size + j
                img_path = data_sets.image_files[index]
                img_name = img_path[img_path.rfind("/")+1:]
                pt_bboxes = predictions[j].cpu().numpy()
                gt_bboxes = labels[labels[:, 0] == j]                                   # 维度是 [K, 6], [img_id, class_id, x, y, w, h]
                gt_bboxes = convert_gt_bboxes_format(gt_bboxes).cpu().numpy()           # 维度是 [K, 6], [left, top, right, bottom, 0, clas_id]

                if not os.path.exists(os.path.join(save_gt_path, img_name.replace(".jpg", ".json"))):
                    save_one_json(os.path.join(save_gt_path, img_name.replace(".jpg", ".json")), gt_bboxes)
                    
                save_one_json(os.path.join(save_pt_path, img_name.replace(".jpg", ".json")), pt_bboxes)
            
                if index == rand_i:
                    show_gt_pt_dict["image"] = tmp_img[j]
                    show_gt_pt_dict["gt_bboxes"] = gt_bboxes
                    show_gt_pt_dict["pt_bboxes"] = pt_bboxes
            
        return save_gt_path, save_pt_path, show_gt_pt_dict



# 根据训练的模型，评估数据集，计算mAP
def estimate(model, val_img_txt, prefix, label_map, image_size=640, batch_size=16, num_workers=16,
                                   nms_max_output_det=30000, nms_thres=0.5, conf_thres=0.001, device="cuda"):

    # print("Starting evalate validation dataset...")
    model.eval()
    
    
    gt_json_dir, pt_json_dir, show_gt_pt_dict = generate_gt_pt_json(model, val_img_txt, prefix, 
                                                                    img_size=image_size,
                                                                    batch_size=batch_size,
                                                                    num_workers=num_workers,
                                                                    max_det=nms_max_output_det,
                                                                    nms_thres=nms_thres,
                                                                    conf_thres=conf_thres,
                                                                    device=device)
    
    # 展示预测结果和标注的区别
    show_image = show_compare_image(show_gt_pt_dict, label_map=label_map)       # 这里的label_map是一个类别列表
    
    # 使用COCO API 开始计算mAP
    coco_tool = COCOmAP(gt_json_dir, pt_json_dir, label_map=label_map, prefix=prefix)
    
    print("num_gt: ", coco_tool.num_gt)
    print("num_pt: ", coco_tool.num_pt)
    
    mAP, AP50, AP75 = coco_tool.evaluate()
    
    
    return mAP, AP50, AP75, show_image







if __name__ == "__main__":
    
    val_img_txt = "/workspace/datasets/PASCAL_VOC2007/VOC2007_trainval/debug.txt"
    
    from models.yolo import YoloV5
    
    # model = YoloV5(yaml_cfg_file="/workspace/yolov5-pro/models/yolov5s.yaml").cuda()
    
    
    # gt_json_dir, pt_json_dir, show_img = generate_gt_pt_json(model, val_img_txt, prefix="debug")

    # generate_coco_json(gt_json_dir, pt_json_dir, None, "debug")
    

    a = Path("/root/aaa/bbbb/ccc/ddd").parent

    print(a)
























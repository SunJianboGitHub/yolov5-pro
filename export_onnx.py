#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   export_onnx.py
@Time    :   2022/11/11 10:52:50
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   Convert Pytorch model to ONNX.
'''


from models.yolo import YoloV5
from models.common import *



model = YoloV5(1, "/workspace/yolov5-pro/models/yolov5s-v2.yaml").cuda()





def convert_onnx():
    pass






if __name__ == "__main__":
    
    for m in model.model.modules():
        if isinstance(m, Focus):
            m.forward = m.forward_export
            print("focus")
    pass



































#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   general.py
@Time    :   2022/09/27 16:04:42
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   some general functions
'''

import os 
import cv2
import sys
import platform
import math
import torch
import random
import logging
import shutil
import datetime
import numpy as np
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def setup_random_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False










def make_divisible(x, divisor):                         # 返回一个整除divisor的整数x，向上取整
    return math.ceil(x / divisor) * divisor


def mkdirs(directory):
    try:
        os.makedirs(directory)
    except Exception as e:
        ...

def mkparents(path):
    parent = Path(path).parent
    if not os.path.exists(parent):
        mkdirs(parent)





def build_logger(path, log_name="yolov5"):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    mkparents(path)

    rf_handler = logging.handlers.TimedRotatingFileHandler(path, when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
    formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)d][%(asctime)s]: %(message)s')
    rf_handler.setFormatter(formatter)
    logger.addHandler(rf_handler)

    sh_handler = logging.StreamHandler()
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    return logger

def build_default_logger():
    logger = logging.getLogger("DefaultLogger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s][%(filename)s:%(lineno)d][%(asctime)s]: %(message)s')
    sh_handler = logging.StreamHandler()
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    return logger


def copy_code_to(src, dst):
    if len(dst) == 0 or dst == ".":
        print("invalid operate, copy to current directory")
        return

    for file in os.listdir(src):
        if file.endswith(".py"):
            source = f"{src}/{file}"
            dest = f"{dst}/{file}"
            mkparents(dest)
            shutil.copy(source, dest)


# 单例模式
class SingleInstanceLogger:
    def __init__(self):
        self.logger = build_default_logger()

    def __getattr__(self, name):
        return getattr(self.logger, name)


def setup_single_instance_logger(path):
    global _single_instance_logger
    _single_instance_logger.logger = build_logger(path)

_single_instance_logger = SingleInstanceLogger()





























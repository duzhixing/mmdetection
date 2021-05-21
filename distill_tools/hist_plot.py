import argparse
import cv2
import numpy as np

import matplotlib.pyplot as plt

import torch
import mmcv

from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.apis import inference_detector

from mmdet.apis import init_detector 

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet show grad a model')
    parser.add_argument('--config', default="./configs/distilling_gfl_iou/resnet50_resnet101_gfl_iou_coco.py" ,help='test config file path')
    parser.add_argument('--checkpoint', default="./result_coco/resnet50_resnet101_gfl_iou_coco/epoch_12.pth", help='checkpoint file')
    args = parser.parse_args()
    return args

feat = []
cls_head = []

def feat_hook_forward(model, input, output):
    feat.append(output)

def cls_hook_forward(model, input, output):
    cls_head.append(output)

def evaluate_case(detector, dl):
    loss_dict = dict()
    nan_list = []
    device = next(detector.parameters()).device
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        if isinstance(data["img"], list):
            for key in data.keys():
                data[key] = data[key][0]        
        filename = data['img_metas'].data[0][0]["filename"]
        data = scatter(data, [device])[0]
        with torch.no_grad():
            loss = detector(return_loss=True, **data)
        # if i > 10:
        #     break
        # total_loss = sum([sum(loss[key]).cpu().numpy() for key in loss])
        break
        prog_bar.update()
    
if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.train.pipeline[3]["flip_ratio"] = float(0)
    device = "cuda:2"
    detector = init_detector(cfg, checkpoint=args.checkpoint, device=device)

    for m in detector.neck.fpn_convs:
        m.register_forward_hook(feat_hook_forward)
    for cls_ in detector.:
        cls_.register_forward_hook(cls_hook_forward)
    

    ds = build_dataset(cfg.data.test)
    dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)

    evaluate_case(detector, dl)


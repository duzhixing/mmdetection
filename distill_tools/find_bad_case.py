from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from pathlib import Path

from mmcv.parallel import MMDataParallel

import os
import mmcv
import math
import torch

import argparse
import pickle as pkl
import pdb

# student: /workspace/S/duzhixing/workspace/model/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth
# teacher: /workspace/S/duzhixing/workspace/model/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth
# TP : Lable + MAX:   result/check/resnet50_resnet101_retinanet_label_thr/epoch_24.pth
# TP + FP: MAX:       result/check/resnet50_resnet101_retinanet_max_thr/epoch_24.pth
# TP + FN: Label:     result/check/resnet50_resnet101_retinanet_label/epoch_24.pth
# TP + TN: tp + tn:   result/check/resnet50_resnet101_retinanet_tp_tn/epoch_24.pth

# max_all_101:   result/resnet50_resnet101_retinanet_max/epoch_24.pth
# max_all_x101:  result/resnet50_resnext101_retinanet_max_all/epoch_24.pth

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default="./configs/retinanet/retinanet_r50_fpn_2x_coco.py",help='test config file path')
    parser.add_argument('--checkpoint', default="result/resnet50_resnext101_retinanet_max_all/epoch_24.pth",help='checkpoint file')
    parser.add_argument('--out_dir', default="tmp/analysis", help='output result file in pickle format')
    args = parser.parse_args()
    return args



# def init_detector(config, checkpoint=None, device='cuda:0'):
#     config.model.pretrained = None
#     model = build_detector(config.model, test_cfg=config.test_cfg, train_cfg=config.train_cfg)
#     if checkpoint is not None:
#         map_loc = 'cpu' if device == 'cpu' else None
#         checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
#         model.CLASSES = checkpoint['meta']['CLASSES']
#     model.cfg = config  # save the config in the model for convenience
#     model.to(device)
#     model.eval()
#     return model

def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'),train_cfg=config.get('train_cfg'))
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def draw_bbox_with_gt(detector, img, gt_bboxes, gt_labels, score_thr=0.3, show=True):
    output_name = os.path.join(f"{out_dir}/",
                            Path(img).name)
    preds = inference_detector(detector, img)
    img = detector.show_result(img, preds, score_thr=score_thr, show=False)
    mmcv.imshow_det_bboxes(img,
                           gt_bboxes,
                           gt_labels,
                           show=show,
                           out_file=output_name,
                           bbox_color='blue',
                           text_color='blue',)

def evaluate_case(detector, data_loader):
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
        total_loss = sum(loss["loss_cls"]) + sum(loss["loss_bbox"])
        total_loss = total_loss.cpu().numpy()
        if math.isnan(total_loss) or math.isinf(total_loss):
            nan_list.append(filename)
        else:
            loss_dict[filename] = total_loss
        prog_bar.update()

    print(f"\nmax loss is {max(loss_dict.values())}, average loss is {sum(loss_dict.values())/len(loss_dict)}")
    print(f"numbers of nan loss: {len(nan_list)}")
    with open(f"{out_dir}/max_x101.pkl", "wb") as f:
        pkl.dump({"loss": loss_dict, "nan_list": nan_list}, f)

def save_pkl(args):
    cfg = Config.fromfile(args.config)
    # detector = init_detector(cfg.deepcopy(), checkpoint=args.checkpoint, device="cuda:0")
    cfg.data.test.pipeline.insert(1, dict(type='LoadAnnotations', with_bbox=True))
    cfg.data.test.pipeline[-1]["transforms"][-1]["keys"].extend(['gt_bboxes', 'gt_labels'])
    
    detector = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    detector = MMDataParallel(detector, device_ids=[0])

    ds = build_dataset(cfg.data.test)
    dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    evaluate_case(detector, dl)

if __name__ == "__main__":

    args = parse_args()
    global out_dir
    out_dir = args.out_dir
    save_pkl(args)
    print(out_dir)
    # with open(f"tmp/analysis_with_loss.pkl", "rb") as f:
    #     data = pkl.load(f)
    # # print("data = ", data)
    # import matplotlib.pyplot as plt
    # #
    # # loss直方图
    # plt.hist(data["loss"].values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.savefig("loss_hist.jpg")

    # import numpy as np
    # mean, var = np.mean(list(data["loss"].values())), np.var(list(data["loss"].values()))
    # print(mean, var, mean + 20*var)
    # visualize_list = data["nan_list"].copy()
    # for key, value in data['loss'].items():
    #     if value > mean + 40*var:
    #         visualize_list.append(key)
    
    # print(len(visualize_list))
    # # breakpoint()

    # cfg = Config.fromfile(args.config)
    # cfg.data.test.pipeline.insert(1, dict(type='LoadAnnotations', with_bbox=True))
    # cfg.data.test.pipeline[-1]["transforms"][-1]["keys"].extend(['gt_bboxes', 'gt_labels'])
    # raw_data_cfg = cfg.data.test.deepcopy()
    # raw_data_cfg.pipeline = raw_data_cfg.pipeline[:2] + \
    #                         [dict(type='Collect',
    #                               keys=['img', 'gt_bboxes', 'gt_labels'],
    #                               meta_keys=["filename"])]
    # print(raw_data_cfg.pipeline)

    # cfg = Config.fromfile(args.config)
    # detector = init_detector(cfg.deepcopy(), checkpoint=args.checkpoint, device="cuda:0")

    # for item in build_dataset(raw_data_cfg):
    #     filename = item['img_metas'].data['filename']
    #     if filename in visualize_list:
    #         print(filename, data["loss"].get(filename, float("nan")))
    #         draw_bbox_with_gt(detector, filename, item['gt_bboxes'], item['gt_labels'], show=False)




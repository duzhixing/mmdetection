from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
from .distilling_single import Distilling_Single
# from mmdet.apis.inference import init_detector
import cv2
import matplotlib.pyplot as plt 
import numpy as np

@DETECTORS.register_module()
class Distilling_Max(Distilling_Single):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Max, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, distill)
        self.nrows, self.ncols = 1, 5
        self.fig, self.axs = plt.subplots(self.nrows, self.ncols, figsize=(self.ncols * 7, self.nrows * 4))
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        stu_feature_adap = self.stu_feature_adap(x)
        y = self.teacher.extract_feat(img)

        stu_bbox_outs = self.bbox_head(x)
        stu_cls_score = stu_bbox_outs[0]
        stu_bbox_pred = stu_bbox_outs[1]

        tea_bbox_outs = self.teacher.bbox_head(y)
        tea_cls_score = tea_bbox_outs[0]
        tea_bbox_pred = tea_bbox_outs[1]

        layers = len(stu_cls_score)
        distill_feat_loss, distill_cls_loss,  distill_bbox_loss= 0, 0, 0
        self.feat_channel = x[0].shape[1]
        self.cls_channel = stu_cls_score[0].shape[1]
        self.bbox_channel = stu_bbox_pred[0].shape[1]

        mask_all= []
        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values

            mask_all.append(mask)

            focal_mask = mask.detach()
            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')
            bbox_loss = torch.pow((tea_bbox_pred[layer] - stu_bbox_pred[layer]), 2)

            distill_feat_loss += (feat_loss * focal_mask[:,None,:,:]).sum() / focal_mask.sum()
            distill_cls_loss +=  (cls_loss * focal_mask[:,None,:,:]).sum() / focal_mask.sum()
            distill_bbox_loss +=  (bbox_loss * focal_mask[:,None,:,:]).sum() / focal_mask.sum()

        # batchs = len(y[0])
        # for batch in range(batchs):
        #     filename = img_metas[batch]['filename'].split('/')[-1].split('.')[0]
        #     for ax, layer in zip(self.axs, range(layers)):
        #         feat_layer = y[layer].permute(0,2,3,1)
        #         mask_layer = mask_all[layer]
        #         mask_img = self.visualize_cam(mask_layer[batch], img[batch])
        #         # ax.set_title(self.hist_name[ix])
        #         ax.imshow(mask_img)
        #     # plt.savefig(f"tmp/nips/mask/tmp.jpg")
        #     # breakpoint()
        #     plt.savefig(f"tmp/nips/mask/{self.iter}_{filename}.jpg")
        #     plt.cla()

        self.target_list.clear()

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        distill_cls_loss = distill_cls_loss * self.distill_cls_weight
        distill_bbox_loss = distill_bbox_loss * self.distill_bbox_weight

        if self.debug:
            # if self._inner_iter == 10:
            #     breakpoint()
            print(self._inner_iter, distill_feat_loss, distill_cls_loss, distill_bbox_loss)

        if self.distill_warm_step > self.iter:
            distill_feat_loss = (self.iter / self.distill_warm_step) * distill_feat_loss
            distill_cls_loss = (self.iter / self.distill_warm_step) * distill_cls_loss
            distill_bbox_loss = (self.iter / self.distill_warm_step) * distill_bbox_loss

        if self.distill_feat_weight:
            losses.update({"distill_feat_loss":distill_feat_loss})
        if self.distill_cls_weight:
            losses.update({"distill_cls_loss":distill_cls_loss})
        if self.distill_bbox_weight:
            losses.update({"distill_bbox_loss":distill_bbox_loss})

        return losses
    
    def visualize_cam(self, mask, img):
        h, w = img.shape[1:]
        mask = cv2.resize(mask.cpu().numpy(), (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap
        heatmap = torch.stack([r, g, b])
        result = (img - img.min()) / (img.max() - img.min())
        result = heatmap + result.cpu()
        result = result.permute(1, 2, 0)
        result = result.numpy()
        return result

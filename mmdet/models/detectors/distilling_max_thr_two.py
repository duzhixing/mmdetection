from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
from .distilling_two import Distilling_Two
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Max_Thr_Two(Distilling_Single):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Max_Thr_Two, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, distill)
        # self.distill_bbox_weight = distill.get("distill_bbox_weight",0)
        self.thr = distill.mask_thr
    
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

        tea_bbox_outs = self.teacher.bbox_head(y)
        tea_cls_score = tea_bbox_outs[0]

        layers = len(stu_feature_adap)
        distill_feat_loss, distill_cls_loss,  distill_bbox_loss= 0, 0, 0

        for layer in range(layers):
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values

            mask = (mask > self.thr).float()
            focal_mask = mask.detach()
            norms = (focal_mask.sum() + 1e-6)

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            distill_feat_loss += (feat_loss * focal_mask[:,None,:,:]).sum() / norms

        self.target_list.clear()

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight

        if self.debug:
            print(self._inner_iter, distill_feat_loss)

        if self.distill_warm_step > self.iter:
            distill_feat_loss = (self.iter / self.distill_warm_step) * distill_feat_loss

        if self.distill_feat_weight:
            losses.update({"distill_feat_loss":distill_feat_loss})

        return losses

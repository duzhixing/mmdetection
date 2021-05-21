from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
from .distilling_single import Distilling_Single
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Fcos(Distilling_Single):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Fcos, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, distill)
        # self.distill_bbox_weight = distill.get("distill_bbox_weight",0)
    
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
        tea_center_score = tea_bbox_outs[2]

        layers = len(stu_cls_score)
        distill_feat_loss, distill_cls_loss,  distill_bbox_loss= 0, 0, 0
        self.feat_channel = x[0].shape[1]
        self.cls_channel = stu_cls_score[0].shape[1]
        self.bbox_channel = stu_bbox_pred[0].shape[1]

        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            tea_center_score_sigmoid = tea_center_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid * tea_center_score_sigmoid, dim=1).values
            focal_mask = mask.detach()

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')
            bbox_loss = torch.pow((tea_bbox_pred[layer] - stu_bbox_pred[layer]), 2)

            distill_feat_loss += (feat_loss * focal_mask[:,None,:,:]).sum() / focal_mask.sum()
            distill_cls_loss +=  (cls_loss * focal_mask[:,None,:,:]).sum() / focal_mask.sum()
            distill_bbox_loss +=  (bbox_loss * focal_mask[:,None,:,:]).sum() / focal_mask.sum()
            # breakpoint()

        self.target_list.clear()

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        distill_cls_loss = distill_cls_loss * self.distill_cls_weight
        distill_bbox_loss = distill_bbox_loss * self.distill_bbox_weight

        if self.debug:
            # breakpoint()
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

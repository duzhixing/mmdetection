from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
from .distilling_two import Distilling_Two
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Two_Feat(Distilling_Two):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Two_Feat, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            distill=distill)
        self.distill_kind = distill.kind
        self.thr = distill.get("mask_thr",0)

        self.rpn_head.loss_cls.register_forward_hook(hook=self.get_target_hook)
        self.target_list = []
            
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        
        stu_feature_adap = self.stu_feature_adap(x)
        y = self.teacher.extract_feat(img)

        tea_bbox_outs = self.teacher.rpn_head(y)
        tea_cls_score = tea_bbox_outs[0]

        layers = len(stu_feature_adap)
        distill_feat_loss = 0

        for layer in range(layers):
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            
            n, h, w = mask.shape
            target = self.target_list[layer].reshape(n, h, w, -1) != self.rpn_head.num_classes
            target = torch.max(target, dim=-1).values
            mask_thr = (mask > self.thr).float()
            if  self.distill_kind == "max":
                feat_mask = mask
            elif self.distill_kind == "max_thr":
                feat_mask = max_thr
            elif self.distill_kind == "label_max_thr":
                feat_mask = (target * mask_thr).detach()
            elif self.distill_kind == "label":
                feat_mask = target
            norms = (feat_mask.sum() + 1e-6)

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            distill_feat_loss += (feat_loss * feat_mask[:,None,:,:]).sum() / norms
            
        self.target_list.clear()

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight

        if self.debug:
            print(self._inner_iter, distill_feat_loss)

        if self.distill_warm_step > self.iter:
            distill_feat_loss = (self.iter / self.distill_warm_step) * distill_feat_loss

        if self.distill_feat_weight:
            losses.update({"distill_feat_loss":distill_feat_loss})

        return losses

    def get_target_hook(self, module, fea_in, fea_out):
        if type(fea_in[1]) is tuple:
            self.target_list.append(fea_in[1][0])
        else:
            self.target_list.append(fea_in[1])
        return None
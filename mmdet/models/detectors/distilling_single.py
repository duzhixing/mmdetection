from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_Single(SingleStageDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Single, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)
        from mmdet.apis.inference import init_detector

        self.device = torch.cuda.current_device()
        # breakpoint()
        self.teacher = init_detector(distill.teacher_cfg, \
                        distill.teacher_model_path, self.device)
        self.stu_feature_adap = build_neck(distill.stu_feature_adap)

        self.distill_feat_weight = distill.get("distill_feat_weight",0)
        self.distill_cls_weight = distill.get("distill_cls_weight",0)
        self.distill_bbox_weight = distill.get("distill_bbox_weight",0)

        for m in self.teacher.modules():
            for param in m.parameters():
                param.requires_grad = False
        self.distill_warm_step = distill.distill_warm_step
        self.debug = distill.get("debug",False)

        self.bbox_head.loss_cls.register_forward_hook(hook=self.get_target_hook)
        self.target_list = []

        # self.teacher_weight_sum = 0
        # for m in self.teacher.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         self.teacher_weight_sum += torch.sum(m.weight)

    
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
        # self.feat_channel = x[0].shape[1]
        self.cls_channel = stu_cls_score[0].shape[1]
        self.bbox_channel = stu_bbox_pred[0].shape[1]

        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            
            n, h, w = mask.shape
            target = self.target_list[layer].reshape(n, h, w, -1) != self.bbox_head.num_classes
            target = torch.max(target, dim=-1).values
            # mask = mask / mask.max()
            # n, h, w = mask.shape
            # target = self.target_list[layer].reshape(n, h, w, -1) != self.bbox_head.num_classes
            # student_score = torch.max(stu_cls_score_sigmoid.permute(0, 2, 3, 1).reshape(n, h, w, -1, self.bbox_head.num_classes), dim=-1)[0]
            # focal_mask = torch.mean((student_score-target.float())**2 , dim=-1) * mask  # (n, h, w)
            # focal_mask = focal_mask.detach()  # stable
            focal_mask = (mask * target).detach()
            norms = (focal_mask.sum() + 1e-6)

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')
            # bbox_loss = torch.abs((tea_bbox_pred[layer] / tea_bbox_pred[layer].pow(2).sum() \
            #                         - stu_bbox_pred[layer] / stu_bbox_pred[layer].pow(2).sum()))
            bbox_loss = torch.pow((tea_bbox_pred[layer] - stu_bbox_pred[layer]), 2)

            # bbox_loss = (tea_bbox_pred[layer] - stu_bbox_pred[layer]).abs()

            distill_feat_loss += (feat_loss * focal_mask[:,None,:,:]).sum() / norms
            distill_cls_loss +=  (cls_loss * focal_mask[:,None,:,:]).sum() / (norms * self.cls_channel)
            distill_bbox_loss +=  (bbox_loss * focal_mask[:,None,:,:]).sum() / (norms * self.bbox_channel)
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
    
    def get_target_hook(self, module, fea_in, fea_out):
        if type(fea_in[1]) is tuple:
            self.target_list.append(fea_in[1][0])
        else:
            self.target_list.append(fea_in[1])
        return None

import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import pdb
from .distilling_single import Distilling_Single
@DETECTORS.register_module()
class Distilling_Single_IOU(Distilling_Single):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_Single_IOU, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, distill)
        self.anchor_num = distill.anchor_num

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

        batch_size = len(img_metas)
        layers = len(stu_feature_adap)
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, img_metas, device=self.device)

        distill_feat_loss = 0

        for layer in range(layers):
            height = featmap_sizes[layer][0]
            width = featmap_sizes[layer][1]
            mask_batch = []
            for i in range(batch_size):
                IOU_map = self.bbox_overlaps_batch(anchor_list[i][layer], gt_bboxes[i].unsqueeze(0)).view(
                    height, width, self.anchor_num, gt_bboxes[i].shape[0])
                max_iou, _ = torch.max(IOU_map.view(height * width * self.anchor_num,
                                                        gt_bboxes[i].shape[0]), dim=0)
                mask_per_im = torch.zeros([height, width], dtype=torch.int64).cuda()
                for k in range(gt_bboxes[i].shape[0]):
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    max_iou_per_gt = max_iou[k] * 0.5
                    mask_per_gt = torch.sum(IOU_map[:, :, :, k] > max_iou_per_gt,
                                            dim=2)
                    mask_per_im += mask_per_gt

                mask_batch.append(mask_per_im)
            mask_list = []
            for mask in mask_batch:
                mask = (mask > 0).float().unsqueeze(0)
                mask_list.append(mask)
            mask_batch = torch.stack(mask_list, dim=0)
            norms = mask_batch.sum().detach()

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            distill_feat_loss += (feat_loss * mask_batch).sum() / norms

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        self.target_list.clear()
        if self.debug:
            print(self._inner_iter, distill_feat_loss)

        if self.distill_warm_step > self.iter:
            distill_feat_loss = (self.iter / self.distill_warm_step ) * distill_feat_loss
        
        losses.update({"distill_feat_loss":distill_feat_loss})

        return losses

    def bbox_overlaps_batch(self, anchors, gt_boxes):
        """
        anchors: (N, 4) ndarray of float
        gt_boxes: (b, K, 5) ndarray of float
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        batch_size = gt_boxes.size(0)

        if anchors.dim() == 2:

            N = anchors.size(0)
            K = gt_boxes.size(1)

            anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
            gt_boxes = gt_boxes[:, :, :4].contiguous()

            gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
            gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
            gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

            anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
            anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
            anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

            gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
            anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

            boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
            query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

            iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
                torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
            iw[iw < 0] = 0

            ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
                torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
            ih[ih < 0] = 0
            ua = anchors_area + gt_boxes_area - (iw * ih)
            overlaps = iw * ih / ua
            #pdb.set_trace()

            # mask the overlap here.
            overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
            overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

        elif anchors.dim() == 3:
            N = anchors.size(1)
            K = gt_boxes.size(1)

            if anchors.size(2) == 4:
                anchors = anchors[:, :, :4].contiguous()
            else:
                anchors = anchors[:, :, 1:5].contiguous()

            gt_boxes = gt_boxes[:, :, :4].contiguous()

            gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
            gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
            gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

            anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
            anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
            anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

            gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
            anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

            boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
            query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

            iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
                torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
            iw[iw < 0] = 0

            ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
                torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
            ih[ih < 0] = 0
            ua = anchors_area + gt_boxes_area - (iw * ih)

            overlaps = iw * ih / ua

            # mask the overlap here.
            overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
            overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
        else:
            raise ValueError('anchors input dimension is not correct.')

        return overlaps
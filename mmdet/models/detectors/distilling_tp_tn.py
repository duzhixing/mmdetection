from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
from .distilling_single import Distilling_Single
# from mmdet.apis.inference import init_detector
import matplotlib.pyplot as plt 
import math

import cv2
import numpy as np


@DETECTORS.register_module()
class Distilling_TP_TN(Distilling_Single):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_TP_TN, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, distill)
        self.thr = distill.mask_thr

        self.test_iter = 30
        self.bins = 500
        self.tp = torch.zeros(self.bins)
        self.fn = torch.zeros(self.bins)
        self.fp = torch.zeros(self.bins)
        self.tn = torch.zeros(self.bins)
        self.hist_name = ["tp", "fn", "fp", "tn"]
        # self.hist_name = ["tp", "fn", "mask", "fp", "tn",  "fig"]
        self.nrows, self.ncols = 1, 4
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

        tea_bbox_outs = self.teacher.bbox_head(y)
        tea_cls_score = tea_bbox_outs[0]

        layers = len(stu_feature_adap)
        distill_feat_loss, distill_cls_loss,  distill_bbox_loss= 0, 0, 0
        for layer in range(layers):
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            
            n, h, w = mask.shape
            target = self.target_list[layer].reshape(n, h, w, -1) != self.bbox_head.num_classes
            target = torch.max(target, dim=-1).values
            target = target.bool()
            mask_thr = (mask > self.thr)

            focal_mask = (~(target ^ mask_thr)).float()
            focal_mask = focal_mask.detach()
            norms = (focal_mask.sum() + 1e-6)

            feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            distill_feat_loss += (feat_loss * focal_mask[:,None,:,:]).sum() / norms

            tp = mask_thr.float() * target.float()
            fn = (1 - mask_thr.float()) * target.float()
            fp = mask_thr.float() * (1 - target.float())
            tn = (1 - mask_thr.float()) * (1 - target.float())
            zero = torch.zeros_like(tn)
            hist_ = [tp, fn, fp, tn, mask]
            # hist_ = [tp, fn, mask, fp, tn,  zero]
            batchs = 2
            if layer < 2:
                for batch in range(batchs):
                    filename = img_metas[batch]['filename'].split('/')[-1].split('.')[0]
                    diff = self.svd(y[layer][batch]).view(tp[0].shape)
                    fig, axs = plt.subplots(1, 6, figsize=(self.ncols * 7, self.nrows * 4))
                    for ax, ix in zip(axs.flat, range(6)):
                        print(ix)
                        if ix == 5:
                            mask = diff
                        else:
                            mask = hist_[ix][batch]
                        ax.imshow(mask.detach().cpu().numpy())
                    fig.savefig(f"tmp/nips/svd/{self.iter}_{filename}_{layer}.jpg")
                    
            feat_layer = y[layer].permute(0,2,3,1)
            # if self._inner_iter == 4 or True:
            #     self.plot_single_entropy(hist_, feat_layer, layer)
            # if self._inner_iter >= 17:
            #     for batch in range(batchs):
            #         if "000000423944" in img_metas[batch]['filename'] or True:
            #             # print("!!!!!!!!!!!!!!!!", self.iter)
            #             filename = img_metas[batch]['filename'].split('/')[-1].split('.')[0]
            #             ix = 0
            #             for ax, hist in zip(self.axs.flat, hist_):
            #                 mask_img = self.visualize_cam(hist[batch], img[batch])
            #                 ax.set_title(self.hist_name[ix])
            #                 ax.imshow(mask_img)
            #                 ix += 1
            #             plt.savefig(f"tmp/test_mask/{self.iter}_{filename}_{layer}.jpg")
            #             plt.cla()

        #     self.get_entropy(hist_, feat_layer)
        # # self.plot_all_entropy()
        # self.plot_batch_entropy()

        self.target_list.clear()

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight

        if self.debug:
            self.debug_func(distill_feat_loss, self.hist_name)

        if self.distill_warm_step > self.iter:
            distill_feat_loss = (self.iter / self.distill_warm_step) * distill_feat_loss

        # if self.distill_feat_weight:
        #     losses.update({"distill_feat_loss":distill_feat_loss})

        return losses

    def debug_func(self, distill_feat_loss, hist_name):
        print(self._inner_iter, distill_feat_loss)
        # breakpoint()
        if self._inner_iter == self.test_iter:
            # self.plot_all_entropy()
            breakpoint()
            # pass
        #     exit()

    def get_entropy(self, hist_, feat_layer):
        ix = 0
        for ax, hist in zip(self.axs.flat, hist_):
            tmp = hist.bool()
            feat_ = feat_layer[tmp].view(-1)
            feat_ = feat_.detach().cpu().numpy()
            res = ax.hist(feat_, bins=self.bins, range=(-5,5))
            m = getattr(self, self.hist_name[ix])
            m += torch.Tensor(res[0])
            ix += 1
        plt.cla()

    def plot_single_entropy(self, hist_, feat_layer, layer):
        ix = 0
        name = ["TP","FN", "FP", "TN"]
        self.nrows, self.ncols = 1, 4
        fig, axs = plt.subplots(self.nrows, self.ncols, figsize=(self.ncols * 6, self.nrows * 4))
        for ax, hist in zip(axs.flat, hist_[0:2] + hist_[3:5]):
            tmp = hist.bool()
            feat_ = feat_layer[tmp].view(-1)
            feat_ = feat_.detach().cpu().numpy()
            res = ax.hist(feat_, bins=self.bins, range=(-5,5))
            val = torch.Tensor(res[0])
            if val.sum():
                val = val / val.sum()
                val = val[val != 0]
                entropy = (- val * val.log()).sum().item()
            else:
                entropy = 0
                
            ax.set_ylabel("Frequency")
            ax.set_xlabel("value")
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.65, 0.85, f"entropy = %.3f"%entropy, transform=ax.transAxes, bbox=props)
            print(f"{name[ix]}  entropy = {entropy}")
            ix += 1
        fig.savefig(f"tmp/nips/entropy/tmp.jpg")
        # fig.savefig(f"tmp/nips/entropy/{self.iter}_{name[ix]}_{layer}.jpg")

        # for hist in hist_[0:2] + hist_[3:5]:
        #     tmp = hist.bool()
        #     feat_ = feat_layer[tmp].view(-1)
        #     feat_ = feat_.detach().cpu().numpy()
        #     res = plt.hist(feat_, bins=self.bins, range=(-5,5))
        #     val = torch.Tensor(res[0])
        #     if val.sum():
        #         val = val / val.sum()
        #         val = val[val != 0]
        #         entropy = (- val * val.log()).sum().item()
        #     else:
        #         entropy = 0
        #     print(f"{name[ix]}")
                
        #     plt.ylabel("Frequency")
        #     plt.xlabel("value")
        #     # plt.title(f"{name[ix]}")
        #     print(f"{name[ix]}  entropy = {entropy}")
        #     plt.savefig(f"tmp/nips/entropy/{self.iter}_{name[ix]}_{layer}.jpg")
        #     ix += 1
        #     plt.cla()

    def plot_all_entropy(self):

        for ax, hist in zip(self.axs.flat, self.hist_name):
            m = getattr(self, hist)
            if m.sum():
                average = m / m.sum()
                average = average[average != 0]
                entropy = (- average * average.log()).sum().item()
            else:
                entropy = 0
            ax.bar(list(range(self.bins)),m.numpy(),width=1)
            ax.set_ylabel("Frequency")
            ax.set_xlabel("value")
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.65, 0.85, f"entropy = %.3f"%entropy, transform=ax.transAxes, bbox=props)
            
            print(f"{hist}_entropy = %.3f"%entropy)
        # self.fig.savefig(f"tmp/hist/all_{self.test_iter}.jpg")
        self.fig.savefig(f"tmp/nips/entropy/all_{self.iter}.jpg")
        plt.cla()

    def plot_batch_entropy(self):

        for ax, hist in zip(self.axs.flat, self.hist_name):
            m = getattr(self, hist)
            if m.sum():
                average = m / m.sum()
                average = average[average != 0]
                entropy = (- average * average.log()).sum().item()
            else:
                entropy = 0
            ax.bar(np.arange(-5, 5, 10 / self.bins),m.numpy(),width=10 / self.bins)
            # ax.bar(list(range(self.bins)),m.numpy(),width=1)
            ax.set_ylabel("Frequency")
            ax.set_xlabel("value")
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.65, 0.85, f"entropy = %.3f"%entropy, transform=ax.transAxes, bbox=props)
            m -= m
            print(f"{hist}_entropy = %.3f"%entropy)
        self.fig.savefig(f"tmp/nips/entropy/{self.iter}_entropy.jpg")
        self.fig.clear()
        del self.fig
        self.fig, self.axs = plt.subplots(self.nrows, self.ncols, figsize=(self.ncols * 6, self.nrows * 4))
        # self.fig.savefig(f"tmp/nips/entropy/{self.iter}_remove.jpg")
        # breakpoint()
        # self.fig.clear()

    def visualize_cam(self, mask, img):
        h, w = img.shape[1:]
        # mask = mask / 2 
        mask = cv2.resize(mask.cpu().numpy(), (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap
        heatmap = torch.stack([r, g, b])
        # heatmap = heatmap / 2
        # result = (img - img.min())
        result = (img - img.min()) / (img.max() - img.min())
        result = heatmap + result.cpu()
        result = result.permute(1, 2, 0)
        # result = result.div(result.max())
        result = result.numpy()
        return result
    
    def svd(self, feat):
        feat = feat.view(feat.shape[0], -1)
        U, S, V = feat.svd()
        diff = torch.zeros(feat.shape[1])
        for id in range(feat.shape[1]):
            exclude = torch.cat((feat[..., :id], feat[..., id + 1:]), dim=1)
            U, S_, V_ =  exclude.view(exclude.shape[0], -1).svd()
            diff[id] = S_[0] / S_.sum() - S[0] / S.sum()
        return diff

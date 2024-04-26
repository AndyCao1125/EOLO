import numpy as np
import torch
import torch.nn as nn

from utils import box_ops
import sys
from ..backbone import build_backbone
from ..neck import build_neck
from ..basic.conv import Conv 
from ..basic.upsample import UpSample
from ..basic.bottleneck_csp import BottleneckCSP
from ..fusion_module import basic_fusion_module
from spikingjelly.activation_based import functional


class EYOLOTiny(nn.Module):
    def __init__(self, 
                 cfg=None,
                 device=None, 
                 img_size=640, 
                 num_classes=80, 
                 trainable=False, 
                 conf_thresh=0.001, 
                 nms_thresh=0.60,
                 center_sample=False,
                 fusion_method=None,):
        super(EYOLOTiny, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample

        self.fusion_method = fusion_method

        # backbone
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], pretrained=trainable)

        self.event_backbone, event_feature_channels, event_strides = build_backbone(model_name=cfg["event_backbone"], time_step=cfg["time_step"])

        ## Event-RGB Fusion Module   
        self.fusion_s = basic_fusion_module(input_dim=event_feature_channels[0], version=self.fusion_method, time_step=cfg["time_step"])
        self.fusion_m = basic_fusion_module(input_dim=event_feature_channels[1], version=self.fusion_method, time_step=cfg["time_step"])
        self.fusion_l = basic_fusion_module(input_dim=event_feature_channels[2], version=self.fusion_method, time_step=cfg["time_step"])

        self.stride = strides
        anchor_size = cfg["anchor_size"]
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // 3, 2).float()
        self.num_anchors = self.anchor_size.size(1)
        c3, c4, c5 = feature_channels

        # build grid cell
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)

        # head
        self.head_conv_0 = build_neck(model=cfg["neck"], in_ch=c5, out_ch=c5//2)  # 10
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(c4 + c5//2, c4, n=1, shortcut=False)

        # P3/8-small
        self.head_conv_1 = Conv(c4, c4//2, k=1)  # 14
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(c3 + c4//2, c3, n=1, shortcut=False)

        # P4/16-medium
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2)
        self.head_csp_2 = BottleneckCSP(c3 + c4//2, c4, n=1, shortcut=False)

        # P8/32-large
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2)
        self.head_csp_3 = BottleneckCSP(c4 + c5//2, c5, n=1, shortcut=False)

        # det conv
        self.head_det_1 = nn.Conv2d(c3, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(c4, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(c5, self.num_anchors * (1 + self.num_classes + 4), 1)

        if self.trainable:
            # init bias
            self.init_bias()


    def init_bias(self):               
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.head_det_1.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_2.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_3.bias[..., :self.num_anchors], bias_value)


    def create_grid(self, img_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = img_size, img_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh


    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)


    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int64)     ## np.int --> np.int64 for updated numpy version >= 1.20.0
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @torch.no_grad()
    def inference_single_image(self, x, event):
        KA = self.num_anchors
        C = self.num_classes
        # backbone
        # c3, c4, c5 = self.backbone(x)
        img_c3, img_c4, img_c5 = self.backbone(x)

        # event backbone
        event_c3, event_c4, event_c5 = self.event_backbone(event)
        if self.cfg["event_backbone"] == 'spike_r18_jelly' or self.cfg["event_backbone"] == 'spike_r18_jelly_row':
            event_c3 = event_c3.permute(1,0,2,3,4) ## from [T,N,C,H,W] to [N,T,C,H,W]
            event_c4 = event_c4.permute(1,0,2,3,4)
            event_c5 = event_c5.permute(1,0,2,3,4)

        ## multi-level fusion
        c3 = self.fusion_s(img_c3, event_c3)
        c4 = self.fusion_m(img_c4, event_c4)
        c5 = self.fusion_l(img_c5, event_c5)



        # FPN + PAN
        # head
        c6 = self.head_conv_0(c5)
        c7 = self.head_upsample_0(c6)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        # det
        pred_s = self.head_det_1(c13)[0]
        pred_m = self.head_det_2(c16)[0]
        pred_l = self.head_det_3(c19)[0]

        preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        box_pred_list = []

        for i, pred in enumerate(preds):
            # [KA*(1 + C + 4), H, W] -> [KA*1, H, W] -> [H, W, KA*1] -> [HW*KA, 1]
            obj_pred_i = pred[:KA, :, :].permute(1, 2, 0).contiguous().view(-1, 1)
            # [KA*(1 + C + 4), H, W] -> [KA*C, H, W] -> [H, W, KA*C] -> [HW*KA, C]
            cls_pred_i = pred[KA:KA*(1+C), :, :].permute(1, 2, 0).contiguous().view(-1, C)
            # [KA*(1 + C + 4), H, W] -> [KA*4, H, W] -> [H, W, KA*4] -> [HW, KA, 4]
            reg_pred_i = pred[KA*(1+C):, :, :].permute(1, 2, 0).contiguous().view(-1, KA, 4)
            # txty -> xy
            if self.center_sample:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
            else:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
            # twth -> wh
            wh_pred_i = reg_pred_i[None, ..., 2:].exp() * self.anchors_wh[i]
            # xywh -> x1y1x2y2           
            x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
            x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)[0].view(-1, 4)

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=0)
        cls_pred = torch.cat(cls_pred_list, dim=0)
        box_pred = torch.cat(box_pred_list, dim=0)
        
        # normalize bbox
        bboxes = torch.clamp(box_pred / self.img_size, 0., 1.)

        # scores
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)

        # to cpu
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
        
        ## reset spiking neural network
        functional.reset_net(self.event_backbone)

        return bboxes, scores, cls_inds


    def forward(self, x, event, targets=None):
        if not self.trainable:
            return self.inference_single_image(x, event)
        else:
            B = x.size(0)
            KA = self.num_anchors
            C = self.num_classes
            # backbone
            img_c3, img_c4, img_c5 = self.backbone(x)  ## [N,C,H,W]

            # event backbone
            event_c3, event_c4, event_c5 = self.event_backbone(event) ## [T,N,C,H,W]
            if self.cfg["event_backbone"] == 'spike_r18_jelly':
                event_c3 = event_c3.permute(1,0,2,3,4) ## from [T,N,C,H,W] to [N,T,C,H,W]
                event_c4 = event_c4.permute(1,0,2,3,4)
                event_c5 = event_c5.permute(1,0,2,3,4)

            
            c3 = self.fusion_s(img_c3, event_c3)
            c4 = self.fusion_m(img_c4, event_c4)
            c5 = self.fusion_l(img_c5, event_c5)

            # FPN + PAN
            # head
            c6 = self.head_conv_0(c5)
            c7 = self.head_upsample_0(c6)   # s32->s16
            c8 = torch.cat([c7, c4], dim=1)
            c9 = self.head_csp_0(c8)
            # P3/8
            c10 = self.head_conv_1(c9)
            c11 = self.head_upsample_1(c10)   # s16->s8
            c12 = torch.cat([c11, c3], dim=1)
            c13 = self.head_csp_1(c12)  # to det
            # p4/16
            c14 = self.head_conv_2(c13)
            c15 = torch.cat([c14, c10], dim=1)
            c16 = self.head_csp_2(c15)  # to det
            # p5/32
            c17 = self.head_conv_3(c16)
            c18 = torch.cat([c17, c6], dim=1)
            c19 = self.head_csp_3(c18)  # to det

            # det
            pred_s = self.head_det_1(c13)
            pred_m = self.head_det_2(c16)
            pred_l = self.head_det_3(c19)

            preds = [pred_s, pred_m, pred_l]
            obj_pred_list = []
            cls_pred_list = []
            box_pred_list = []

            for i, pred in enumerate(preds):
                # [B, KA*(1 + C + 4), H, W] -> [B, KA, H, W] -> [B, H, W, KA] ->  [B, HW*KA, 1]
                obj_pred_i = pred[:, :KA, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                # [B, KA*(1 + C + 4), H, W] -> [B, KA*C, H, W] -> [B, H, W, KA*C] -> [B, H*W*KA, C]
                cls_pred_i = pred[:, KA:KA*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
                # [B, KA*(1 + C + 4), H, W] -> [B, KA*4, H, W] -> [B, H, W, KA*4] -> [B, HW, KA, 4]
                reg_pred_i = pred[:, KA*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
                # txty -> xy
                if self.center_sample:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
                else:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
                # twth -> wh
                wh_pred_i = reg_pred_i[..., 2:].exp() * self.anchors_wh[i]
                # xywh -> x1y1x2y2
                x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
                x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
                box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1).view(B, -1, 4)

                obj_pred_list.append(obj_pred_i)
                cls_pred_list.append(cls_pred_i)
                box_pred_list.append(box_pred_i)
            
            obj_pred = torch.cat(obj_pred_list, dim=1)
            cls_pred = torch.cat(cls_pred_list, dim=1)
            box_pred = torch.cat(box_pred_list, dim=1)
            
            # normalize bbox
            box_pred = box_pred / self.img_size

            # compute giou between prediction bbox and target bbox
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)

            # giou: [B, HW,]
            giou_pred, iou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)

            # we set giou as the target of the objectness
            targets = torch.cat([0.5 * (giou_pred[..., None].clone().detach() + 1.0), targets], dim=-1)

            functional.reset_net(self.event_backbone)

            return obj_pred, cls_pred, giou_pred, iou_pred, targets





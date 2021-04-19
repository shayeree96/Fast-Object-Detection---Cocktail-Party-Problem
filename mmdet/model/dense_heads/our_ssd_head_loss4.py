import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead


# TODO: add loss evaluator for SSD
@HEADS.register_module()
class OURSSDHead_Loss4(AnchorHead):

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 background_label=None,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes + 1  # add background class
        self.anchor_generator = build_anchor_generator(anchor_generator)
        num_anchors = self.anchor_generator.num_base_anchors

        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * (num_classes + 1),
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # set sampling=False for archor_target
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.TP = 0
        self.FP = 0
        self.num_GT = 0
        self.num_PRED = 0
        self.stats = torch.zeros((21, 2))
        self.matrix = torch.zeros((21, 21), dtype=torch.long)
        print('we are using loss4 head')
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def unionboxes(self, bbox_gt1_both, bbox_gt2_both):
        '''
        Args:
            bbox_gt1_both: (x1,y1,x2,y2)
        Returns: union
        '''
        x1, _ = torch.min(torch.cat((bbox_gt1_both[:, 0].reshape((-1, 1)), bbox_gt2_both[:, 0].reshape((-1, 1))), dim=1), dim=1)
        y1, _ = torch.min(torch.cat((bbox_gt1_both[:, 1].reshape((-1, 1)), bbox_gt2_both[:, 1].reshape((-1, 1))), dim=1), dim=1)
        x2, _ = torch.max(torch.cat((bbox_gt1_both[:, 2].reshape((-1, 1)), bbox_gt2_both[:, 2].reshape((-1, 1))), dim=1), dim=1)
        y2, _ = torch.max(torch.cat((bbox_gt1_both[:, 3].reshape((-1, 1)), bbox_gt2_both[:, 3].reshape((-1, 1))), dim=1), dim=1)
        gt_new = torch.cat((x1.reshape((-1, 1)), y1.reshape((-1, 1)), x2.reshape((-1, 1)), y2.reshape((-1, 1))), dim=1)
        return gt_new

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        loss_cls_all1 = F.cross_entropy(
            cls_score, labels[0], reduction='none') * label_weights[0]
        loss_cls_all2 = F.cross_entropy(
            cls_score, labels[1], reduction='none') * label_weights[1]
        loss_cls_all = 0.5 * loss_cls_all1 + 0.5 * loss_cls_all2

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = (((labels[0] >= 0) & (labels[1] >= 0)) &
                    ((labels[0] < self.background_label) | (labels[1] < self.background_label))).nonzero().reshape(-1)
        pos_inds_only1 = (((labels[0] >= 0) & (labels[1] >= 0)) &
                    ((labels[0] < self.background_label) & (labels[1] == self.background_label))).nonzero().reshape(-1)
        pos_inds_only2 = (((labels[0] >= 0) & (labels[1] >= 0)) &
                    ((labels[0] == self.background_label) & (labels[1] < self.background_label))).nonzero().reshape(-1)
        pos_inds_both = (((labels[0] >= 0) & (labels[1] >= 0)) &
                    ((labels[0] < self.background_label) & (labels[1] < self.background_label))).nonzero().reshape(-1)
        neg_inds = ((labels[0] == self.background_label) & (labels[1] == self.background_label)).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds_both].sum() + loss_cls_all1[pos_inds_only1].sum() + loss_cls_all2[pos_inds_only2].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        bbox_targets_1 = bbox_targets[0][pos_inds_only1]
        bbox_targets_2 = bbox_targets[1][pos_inds_only2]
        bbox_weights_1 = bbox_weights[0][pos_inds_only1]
        bbox_weights_2 = bbox_weights[1][pos_inds_only2]

        loss_bbox_both = 0.
        loss_bbox_only1 = 0.
        loss_bbox_only2 = 0.
        if len(pos_inds_only1) > 0:
            loss_bbox_only1 = smooth_l1_loss(
                bbox_pred[pos_inds_only1],
                bbox_targets_1,
                bbox_weights_1,
                beta=self.train_cfg.smoothl1_beta,
                avg_factor=num_total_samples)
        if len(pos_inds_only2) > 0:
            loss_bbox_only2 = smooth_l1_loss(
                bbox_pred[pos_inds_only2],
                bbox_targets_2,
                bbox_weights_2,
                beta=self.train_cfg.smoothl1_beta,
                avg_factor=num_total_samples)
        if len(pos_inds_both) > 0:
            bbox_gt1_both = self.bbox_coder.decode(anchor[pos_inds_both], bbox_targets[0][pos_inds_both])
            bbox_gt2_both = self.bbox_coder.decode(anchor[pos_inds_both], bbox_targets[1][pos_inds_both])
            bbox_targets_both = self.unionboxes(bbox_gt1_both, bbox_gt2_both)
            bbox_targets_both = self.bbox_coder.encode(anchor[pos_inds_both], bbox_targets_both)
            bbox_weights_both = bbox_weights[0][pos_inds_both]
            assert bbox_weights[0][pos_inds_both].sum() == bbox_weights[1][pos_inds_both].sum()

            loss_bbox_both = smooth_l1_loss(
                bbox_pred[pos_inds_both],
                bbox_targets_both,
                bbox_weights_both,
                beta=self.train_cfg.smoothl1_beta,
                avg_factor=num_total_samples)

        loss_bbox = loss_bbox_only1 + loss_bbox_only2 + loss_bbox_both
        if (len(pos_inds_only1) + len(pos_inds_only2) + len(pos_inds_both)) == 0:
            print(pos_inds, pos_inds_only1, pos_inds_only2, pos_inds_both)
            return loss_cls[None], 0. * loss_cls[None]
        return loss_cls[None], loss_bbox

    def precision_recall(self, cls_score, target, topk=2, confident=0.05):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            output = F.softmax(cls_score)
            score, pred = output.topk(topk, 1, True, True)
            idx = torch.where(score[:, 1] < confident)[0] # if a top2 confidnent is lower than 0.1 then the prediction is 2*top1
            pred[idx, 1] = pred[idx, 0]
            pred = pred.t()
            feature_num = pred.shape[1]

            FP1 = pred[0, :]
            FP2 = pred[1, :]
            G1 = target[0, :]
            G2 = target[1, :]
            ################################ calculate number of pixel and image for each class #################
            # for i in range(self.stats.shape[0]):
            #    self.stats[i, 0] += torch.sum(G1 == i) + torch.sum(G2 == i)
            #    if torch.sum(G1 == i) != 0: self.stats[i, 1] += 1
            #    if torch.sum(G2 == i) != 0: self.stats[i, 1] += 1
            # print(' stats:\n', self.stats.t())
            condition1_1 = FP1 == G1
            condition1_2 = FP2 == G1
            condition1_3 = G1 != self.background_label
            condition1 = torch.logical_and(torch.logical_or(condition1_1, condition1_2), condition1_3)
            condition2_1 = FP1 == G2
            condition2_2 = FP2 == G2
            condition2_3 = G2 != self.background_label
            condition2 = torch.logical_and(torch.logical_or(condition2_1, condition2_2), condition2_3)
            condition3_1 = FP1 != G1
            condition3_2 = FP1 != G2
            condition3 = condition3_1 & condition3_2
            condition4_1 = FP2 != G1
            condition4_2 = FP2 != G2
            condition4 = condition4_1 & condition4_2

            self.TP += torch.sum(condition1) + torch.sum(condition2)
            self.num_GT += torch.sum(condition1_3) + torch.sum(condition2_3) # Groundtruth that is not background
            self.FP += torch.sum(condition3) + torch.sum(condition4)
            self.num_PRED += feature_num*2

            print('\n'+'recall:', self.TP.float()/self.num_GT.float(), 'precision:', 1 - self.FP.float()/self.num_PRED)

    def loss_single_cocktail(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        self.precision_recall(cls_score, labels, topk=2, confident=0.1)
        return 0, 0

    def confusion_matrix(self, cls_score, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            output = F.softmax(cls_score)
            score, pred = output.topk(1, 1, True, True)
            pred = pred.view(-1)
            target = target[0, :].view(-1)
            for i in range(pred.shape[0]):
                self.matrix[pred[i], target[i]] += 1
            print('\n'+'confusion:', self.matrix,  torch.sum(self.matrix, dim=0),  torch.sum(self.matrix, dim=1))

    def loss_single_singlebatch(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        self.confusion_matrix(cls_score, labels)
        return 0, 0

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        newbatch = int(num_images / 2)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                newbatch, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_labels = all_labels[0:2*newbatch].reshape((2, newbatch, -1)).permute(1, 0, 2)

        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_label_weights = all_label_weights[0:2*newbatch].reshape((2, newbatch, -1)).permute(1, 0, 2)

        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(newbatch, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_targets = all_bbox_targets[0:2*newbatch].reshape((2, newbatch, -1, 4)).permute(1, 0, 2, 3)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = all_bbox_weights[0:2*newbatch].reshape((2, newbatch, -1, 4)).permute(1, 0, 2, 3)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(newbatch):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        # return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_cocktail(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        newbatch = int(num_images / 2)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                newbatch, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_labels = all_labels[0:2*newbatch].reshape((2, newbatch, -1)).permute(1, 0, 2)

        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_label_weights = all_label_weights[0:2*newbatch].reshape((2, newbatch, -1)).permute(1, 0, 2)

        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(newbatch, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_targets = all_bbox_targets[0:2*newbatch].reshape((2, newbatch, -1, 4)).permute(1, 0, 2, 3)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = all_bbox_weights[0:2*newbatch].reshape((2, newbatch, -1, 4)).permute(1, 0, 2, 3)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(newbatch):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        losses_cls, losses_bbox = multi_apply(
            self.loss_single_cocktail,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        # return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        return dict(loss_cls=losses_cls)

    def loss_singlebatch(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        newbatch = int(num_images / 2)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                newbatch, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_labels = all_labels[0:2*newbatch].reshape((2, newbatch, -1)).permute(1, 0, 2)

        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_label_weights = all_label_weights[0:2*newbatch].reshape((2, newbatch, -1)).permute(1, 0, 2)

        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(newbatch, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_targets = all_bbox_targets[0:2*newbatch].reshape((2, newbatch, -1, 4)).permute(1, 0, 2, 3)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = all_bbox_weights[0:2*newbatch].reshape((2, newbatch, -1, 4)).permute(1, 0, 2, 3)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(newbatch):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        losses_cls, losses_bbox = multi_apply(
            self.loss_single_singlebatch,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        # return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        return dict(loss_cls=losses_cls)

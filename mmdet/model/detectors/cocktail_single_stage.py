import torch.nn as nn
import torchvision
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import torch
import numpy as np

@DETECTORS.register_module()
class CCSingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CCSingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(CCSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        minibatch = len(img)
        newbatch = int(minibatch / 2)
        #  if len(img) = 7, newbatch = 3, the last one image is ignored
        if minibatch != 1:
            img_ = img[0:newbatch] + img[newbatch:newbatch+newbatch]
        else:
            img_ = img
            img_metas = [img_metas[0], img_metas[0]]
            gt_bboxes = [gt_bboxes[0], gt_bboxes[0]]
            gt_labels = [gt_labels[0], gt_labels[0]]
        x = self.extract_feat(img_)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    ''' for one image per batch only.
    # Used for testing regular ssd and no_loc_loss ssd
    def simple_test(self, img, img_metas, rescale, gt_bboxes, gt_labels):
        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_singlebatch_test(x, [img_metas[0], img_metas[0]], [gt_bboxes[0], gt_bboxes[0]],
                                              [gt_labels[0], gt_labels[0]], None)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]
    '''
    # '''
    def simple_test(self, img, img_metas, rescale, gt_bboxes, gt_labels):
        # used for cocktail testing
        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]

        minibatch = len(img)
        newbatch = int(minibatch / 2)
        #  if len(img) = 7, newbatch = 3, the last one image is ignored
        if minibatch != 1:
            img_ = img[0:newbatch] + img[newbatch:newbatch + newbatch]
        else:
            img_ = img
            img_metas = [img_metas[0], img_metas[0]]
            gt_bboxes = [gt_bboxes[0], gt_bboxes[0]]
            gt_labels = [gt_labels[0], gt_labels[0]]
        x = self.extract_feat(img_)
        losses = self.bbox_head.forward_cocktail_test(x, img_metas, gt_bboxes,
                                              gt_labels, None)
        # print(losses)
        nonsense = [np.zeros((0, 5)) for i in range(20)]
        return nonsense
    # '''

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

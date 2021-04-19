from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class RetinaNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

@DETECTORS.register_module()
class CCRetinaNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CCRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

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


@DETECTORS.register_module()
class CC9RetinaNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CC9RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

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
            img_ = 0.5 * img[0:newbatch] + 0.5 * img[newbatch:newbatch+newbatch]
        else:
            img_ = img
            img_metas = [img_metas[0], img_metas[0]]
            gt_bboxes = [gt_bboxes[0], gt_bboxes[0]]
            gt_labels = [gt_labels[0], gt_labels[0]]
        x = self.extract_feat(img_)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses


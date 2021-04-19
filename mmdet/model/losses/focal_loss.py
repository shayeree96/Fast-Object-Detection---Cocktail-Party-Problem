import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..builder import LOSSES
from .utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


def CCsigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss1 = _sigmoid_focal_loss(pred, target[0], gamma, alpha)
    loss2 = _sigmoid_focal_loss(pred, target[1], gamma, alpha)
    weight1 = weight[0].view(-1, 1)
    weight2 = weight[1].view(-1, 1)
    loss1 = loss1 * weight1
    loss2 = loss2 * weight2
    loss = 0.5 * loss1 + 0.5 * loss2
    background_label = loss1.shape[1]
    pos_ind_only1 = ((target[0] < background_label) & (target[1] == background_label)).nonzero().reshape(-1)
    pos_ind_only2 = ((target[1] < background_label) & (target[0] == background_label)).nonzero().reshape(-1)
    pos_ind_both = ((target[1] < background_label) & (target[0] < background_label)).nonzero().reshape(-1)
    sig_ind_only1 = target[0][pos_ind_both]
    sig_ind_only2 = target[1][pos_ind_both]
    loss[pos_ind_only1] = loss1[pos_ind_only1]
    loss[pos_ind_only2] = loss2[pos_ind_only2]
    loss[pos_ind_both, sig_ind_only1] = loss1[pos_ind_both, sig_ind_only1]
    loss[pos_ind_both, sig_ind_only2] = loss2[pos_ind_both, sig_ind_only2]

    loss = weight_reduce_loss(loss, None, reduction, avg_factor) # set weight == None since already multiplied
    return loss


@LOSSES.register_module()
class CCFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(CCFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * CCsigmoid_focal_loss(
                pred, # (N, 80)
                target, # (2, N) N = 122400
                weight, # (2, N)
                gamma=self.gamma, # 2.0
                alpha=self.alpha, # 0.25
                reduction=reduction, # 'mean'
                avg_factor=avg_factor) # 325952
        else:
            raise NotImplementedError
        return loss_cls

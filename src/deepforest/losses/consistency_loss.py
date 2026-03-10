import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import ramps


class consistency_weight:
    """ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup',
    'log_rampup', 'exp_rampup']"""

    def __init__(
        self,
        final_w,
        iters_per_epoch,
        rampup_starts=0,
        rampup_ends=7,
        ramp_type="sigmoid_rampup",
    ):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = self.rampup_ends - self.rampup_starts
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(
            cur_total_iter - self.rampup_starts, self.rampup_length
        )
        return self.final_w * self.current_rampup


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    return F.cross_entropy(
        input_logits / temperature, target_targets, ignore_index=ignore_index
    )


# for FocalLoss
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def get_alpha(supervised_loader):
    # get number of classes
    num_labels = 0
    for _image_batch, label_batch in supervised_loader:
        label_batch.data[label_batch.data == 255] = (
            0  # pixels of ignore class added to background
        )
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique), num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for _image_batch, label_batch in supervised_loader:
        label_batch.data[label_batch.data == 255] = (
            0  # pixels of ignore class added to background
        )
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack(
            [(label_batch.data == x_u).sum() for x_u in l_unique]
        )  # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        ignore_index=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
            alpha = 1 / alpha  # inverse of class frequency
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()

        # to resolve error in idx in scatter_
        idx[idx == 225] = 0

        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class abCE_loss(nn.Module):
    """Annealed-Bootstrapped cross-entropy loss."""

    def __init__(
        self,
        iters_per_epoch,
        epochs,
        num_classes,
        weight=None,
        reduction="mean",
        thresh=0.7,
        min_kept=1,
        ramp_type="log_rampup",
    ):
        super().__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type

        if ramp_type is not None:
            self.rampup_func = getattr(ramps, ramp_type)
            self.iters_per_epoch = iters_per_epoch
            self.num_classes = num_classes
            self.start = 1 / num_classes
            self.end = 0.9
            self.total_num_iters = (epochs - (0.6 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter, epoch):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index, curr_iter, epoch):
        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = (
            target.contiguous().view(
                -1,
            )
            != ignore_index
        )
        sort_prob, sort_indices = (
            prob.contiguous()
            .view(
                -1,
            )[mask]
            .contiguous()
            .sort()
        )

        if self.ramp_type is not None:
            thresh = self.threshold(curr_iter=curr_iter, epoch=epoch)
        else:
            thresh = self.thresh

        min_threshold = (
            sort_prob[min(batch_kept, sort_prob.numel() - 1)]
            if sort_prob.numel() > 0
            else 0.0
        )
        threshold = max(min_threshold, thresh)
        loss_matrix = F.cross_entropy(
            predict,
            target,
            weight=self.weight.to(predict.device) if self.weight is not None else None,
            ignore_index=ignore_index,
            reduction="none",
        )
        loss_matirx = loss_matrix.contiguous().view(
            -1,
        )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == "sum" or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == "mean":
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError("Reduction Error!")


def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad and not targets.requires_grad
    assert inputs.size() == targets.size()  # (batch_size * num_classes * H * W)
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction="none")
        mask = targets.max(1)[0] > threshold
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0:
            loss_mat = torch.tensor([0.0]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(
            inputs, targets, reduction="mean"
        )  # take the mean over the batch_size


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad and not targets.requires_grad
    assert inputs.size() == targets.size()

    if use_softmax:
        targets = F.softmax(targets, dim=1)
    if conf_mask:
        loss_mat = F.kl_div(F.log_softmax(inputs, dim=1), targets, reduction="none")
        mask = targets.max(1)[0] > threshold
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0:
            loss_mat = torch.tensor([0.0]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(inputs, targets, reduction="mean")


def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad and not targets.requires_grad
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction="mean")
    kl2 = F.kl_div(torch.log(targets + epsilon), M, reduction="mean")
    return (kl1 + kl2) * 0.5


def pair_wise_loss(unsup_outputs, size_average=True, nbr_of_pairs=8):
    """Pair-wise loss in the sup.

    mat.
    """
    if isinstance(unsup_outputs, list):
        unsup_outputs = torch.stack(unsup_outputs)

    # Only for a subset of the aux outputs to reduce computation and memory
    unsup_outputs = unsup_outputs[torch.randperm(unsup_outputs.size(0))]
    unsup_outputs = unsup_outputs[:nbr_of_pairs]

    temp = torch.zeros_like(unsup_outputs)  # For grad purposes
    for i, u in enumerate(unsup_outputs):
        temp[i] = F.softmax(u, dim=1)
    mean_prediction = temp.mean(0).unsqueeze(0)  # Mean over the auxiliary outputs
    pw_loss = ((temp - mean_prediction) ** 2).mean(0)  # Variance
    pw_loss = pw_loss.sum(1)  # Sum over classes
    if size_average:
        return pw_loss.mean()
    return pw_loss.sum()

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0.0

    def forward(self, img_list, margin=0):
        length = len(img_list)
        self.loss = 0.0
        B, C, H, W = img_list[0].shape
        for i in range(length - 1):
            for j in range(i + 1, length):
                self.loss = self.loss + torch.sum(
                    F.relu(
                        img_list[j].sum(-1).sum(-1).sum(-1)
                        - img_list[i].sum(-1).sum(-1).sum(-1)
                        + margin
                    )
                )

        self.loss = self.loss / (B * length * (length - 1) / 2)
        return self.loss


class RankLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.countloss_criterion = nn.MSELoss(reduction="sum")
        self.rankloss_criterion = MarginRankLoss()
        self.rankloss = 0.0
        self.losses = {}

    def forward(self, unlabeled_results):
        self.rankloss = 0.0
        self.losses = {}

        if unlabeled_results is None:
            self.rankloss = 0.0
        elif isinstance(unlabeled_results, tuple) and len(unlabeled_results) > 0:
            self.rankloss = self.rankloss_criterion(unlabeled_results)
        elif isinstance(unlabeled_results, list) and len(unlabeled_results) > 0:
            count = 0
            for i in range(len(unlabeled_results)):
                if (
                    isinstance(unlabeled_results[i], tuple)
                    and len(unlabeled_results[i]) > 0
                ):
                    temploss = self.rankloss_criterion(unlabeled_results[i])
                    self.losses.update({f"unlabel_{str(i + 1)}_loss": temploss})
                    self.rankloss += temploss

                    count += 1
            if count > 0:
                self.rankloss = self.rankloss / count

        return self.rankloss

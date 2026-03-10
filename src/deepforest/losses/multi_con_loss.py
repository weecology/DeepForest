import torch
import torch.nn as nn

from .consistency_loss import softmax_kl_loss


class MultiConLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.countloss_criterion = nn.MSELoss(reduction="sum")
        self.multiconloss = 0.0
        self.losses = {}

    def forward(self, unlabeled_results):
        self.multiconloss = 0.0
        self.losses = {}

        if unlabeled_results is None:
            self.multiconloss = 0.0
        elif isinstance(unlabeled_results, list) and len(unlabeled_results) > 0:
            count = 0
            for i in range(len(unlabeled_results[0])):
                with torch.set_grad_enabled(False):
                    preds_mean = (
                        unlabeled_results[0][i]
                        + unlabeled_results[1][i]
                        + unlabeled_results[2][i]
                    ) / len(unlabeled_results)
                for j in range(len(unlabeled_results)):
                    var_sel = softmax_kl_loss(unlabeled_results[j][i], preds_mean)
                    exp_var = torch.exp(-var_sel)
                    consistency_dist = (preds_mean - unlabeled_results[j][i]) ** 2
                    temploss = (
                        torch.mean(consistency_dist * exp_var) / (exp_var + 1e-8)
                        + var_sel
                    )

                    self.losses.update({f"unlabel_{str(i + 1)}_loss": temploss})
                    self.multiconloss += temploss

                    count += 1
            if count > 0:
                self.multiconloss = self.multiconloss / count

        return self.multiconloss

from typing import cast

import torch
from torch.nn import Module

from .sinkhorn import sinkhorn_knopp


class OT_Loss(Module):
    def __init__(self, norm_cood, device, num_of_iter_in_ot=100, reg=1.0):
        super().__init__()
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        output_h = normed_density.size(2)
        output_w = normed_density.size(3)

        x_cood = (
            torch.arange(output_w, dtype=torch.float32, device=self.device) + 0.5
        ).unsqueeze(0)
        y_cood = (
            torch.arange(output_h, dtype=torch.float32, device=self.device) + 0.5
        ).unsqueeze(0)
        if self.norm_cood:
            x_cood = x_cood / output_w * 2 - 1
            y_cood = y_cood / output_h * 2 - 1

        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0  # wasserstain distance
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                if self.norm_cood:
                    x = im_points[:, 0].unsqueeze(1) / output_w * 2 - 1
                    y = im_points[:, 1].unsqueeze(1) / output_h * 2 - 1
                else:
                    x = im_points[:, 0].unsqueeze(1)
                    y = im_points[:, 1].unsqueeze(1)
                x_dis = (
                    -2 * torch.matmul(x, x_cood) + x * x + x_cood * x_cood
                )  # [#gt, #cood]
                y_dis = -2 * torch.matmul(y, y_cood) + y * y + y_cood * y_cood
                y_dis.unsqueeze_(2)
                x_dis.unsqueeze_(1)
                dis = y_dis + x_dis
                dis = dis.view((dis.size(0), -1))  # size of [#gt, #cood * #cood]

                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(
                    self.device
                )

                # use sinkhorn to solve OT, compute optimal beta.
                P, log = sinkhorn_knopp(
                    target_prob,
                    source_prob,
                    dis,
                    self.reg,
                    maxIter=self.num_of_iter_in_ot,
                    log=True,
                )
                log = cast(dict[str, torch.Tensor], log)
                beta = log["beta"]  # size is the same as source_prob: [#cood * #cood]
                ot_obj_values += torch.sum(
                    normed_density[idx] * beta.view([1, output_h, output_w])
                )
                # compute the gradient of OT loss to predicted density (unnormed_density).
                # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                im_grad_1 = (
                    (source_count) / (source_count * source_count + 1e-8) * beta
                )  # size of [#cood * #cood]
                im_grad_2 = (source_density * beta).sum() / (
                    source_count * source_count + 1e-8
                )  # size of 1
                im_grad = im_grad_1 - im_grad_2
                im_grad = im_grad.detach().view([1, output_h, output_w])
                # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
                loss += torch.sum(unnormed_density[idx] * im_grad)
                wd += torch.sum(dis * P).item()

        return loss, wd, ot_obj_values

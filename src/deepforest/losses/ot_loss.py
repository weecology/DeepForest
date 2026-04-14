"""PyTorch implementation of Sinkhorn-Knopp for optimal transport.

Based on ot.bregman.sinkhorn from the Python Optimal Transport library
(https://pythonot.github.io), rewritten in Pytorch and taken from the
TreeFormer repository:

Reference: M. Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal
Transport", NeurIPS 2013.
"""

from typing import cast

import torch
from torch.nn import Module

M_EPS = 1e-16


def sinkhorn_knopp(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float = 1e-1,
    maxIter: int = 1000,
    stopThr: float = 1e-9,
    log: bool = False,
    eval_freq: int = 10,
    warm_start: dict | None = None,
) -> tuple[torch.Tensor, dict] | torch.Tensor:
    """Solve entropic-regularized OT via Sinkhorn-Knopp matrix scaling.

    Minimises ``<gamma, C>_F + reg * sum(gamma * log(gamma))``
    subject to ``gamma @ 1 = a`` and ``gamma.T @ 1 = b``.

    Args:
        a:        (na,) source measure (sums to 1).
        b:        (nb,) target measure (sums to 1).
        C:        (na, nb) cost matrix.
        reg:      Entropic regularization strength (> 0).
        maxIter:  Maximum number of Sinkhorn iterations.
        stopThr:  Early-stop threshold on marginal error.
        log:      If True, return ``(P, log_dict)``; otherwise return ``P``.
        eval_freq: Check convergence every this many iterations.
        warm_start: Optional dict with keys ``"u"`` and ``"v"`` to resume
                    from a previous solve.

    Returns:
        P (na, nb) optimal transport plan, and optionally a log dict
        containing ``"u"``, ``"v"``, ``"alpha"``, ``"beta"`` (dual variables)
        and ``"err"`` (list of marginal errors).
    """
    device = a.device
    na, nb = C.shape

    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b doesn't match C"
    assert reg > 0, "reg must be > 0"

    log_dict: dict = {"err": []} if log else {}

    if warm_start is not None:
        u = warm_start["u"]
        v = warm_start["v"]
    else:
        u = torch.ones(na, dtype=a.dtype, device=device) / na
        v = torch.ones(nb, dtype=b.dtype, device=device) / nb

    # Clamp exponent to float32 safe range before exp to prevent underflow.
    K = torch.exp(torch.clamp(C / -reg, min=-80.0))

    it = 1
    err = 1.0
    while err > stopThr and it <= maxIter:
        upre, vpre = u, v
        v = b / (torch.mv(K.t(), u) + M_EPS)
        u = a / (torch.mv(K, v) + M_EPS)

        if torch.any(torch.isnan(u) | torch.isinf(u)) or torch.any(
            torch.isnan(v) | torch.isinf(v)
        ):
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            b_hat = torch.mv(K.t(), u) * v
            err = (b - b_hat).pow(2).sum().item()
            log_dict["err"].append(err)

        it += 1

    if log:
        log_dict["u"] = u
        log_dict["v"] = v
        log_dict["alpha"] = reg * torch.log(u + M_EPS)
        log_dict["beta"] = reg * torch.log(v + M_EPS)
        log_dict["its"] = it

    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return (P, log_dict) if log else P


class OT_Loss(Module):
    def __init__(self, norm_coord, device, num_of_iter_in_ot=100, reg=1.0):
        super().__init__()
        self.device = device
        self.norm_coord = norm_coord
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

    @torch.autocast("cuda", enabled=False)
    @torch.autocast("cpu", enabled=False)
    def forward(self, normed_density, unnormed_density, points):
        # Disable AMP autocast and force float32 to prevent float16
        # underflow in Sinkhorn. K = exp(C / -reg) underflows to zero
        # in float16 for squared distances > ~11, collapsing the
        # transport plan entirely.
        normed_density = normed_density.float()
        unnormed_density = unnormed_density.float()
        points = [p.float() for p in points]

        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        output_h = normed_density.size(2)
        output_w = normed_density.size(3)

        # Define grid coordinates (centered)
        x_cood = (
            torch.arange(output_w, dtype=torch.float32, device=self.device) + 0.5
        ).unsqueeze(0)
        y_cood = (
            torch.arange(output_h, dtype=torch.float32, device=self.device) + 0.5
        ).unsqueeze(0)

        # Optionally normalize coordinates to [-1, 1]
        if self.norm_coord:
            x_cood = x_cood / output_w * 2 - 1
            y_cood = y_cood / output_h * 2 - 1

        loss_terms = []
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0  # Wasserstein distance
        n_active = 0  # Total number of points over all images
        total_its = 0  # Accumulated Sinkhorn iterations

        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                n_active += 1

                # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                if self.norm_coord:
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
                wd += torch.sum(dis * P).item()

                log = cast(dict[str, torch.Tensor], log)
                total_its += log["its"]
                # Clamp beta (dual variable = reg * log(v + eps)) to prevent
                # Sinkhorn divergence from producing inf/nan in the OT loss.
                beta = log["beta"].clamp(min=-1e4, max=1e4)  # [#cood * #cood]
                ot_obj_values = ot_obj_values + torch.sum(
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
                loss_terms.append(torch.sum(unnormed_density[idx] * im_grad))

        if n_active > 0:
            loss = torch.stack(loss_terms).sum() / n_active
            ot_obj_values = ot_obj_values / n_active
        else:
            # All images in this batch have zero points. Keep loss connected
            # to unnormed_density so DDP gradient buckets fire on every rank;
            # the 0.0 multiplier means no actual gradient flows.
            loss = 0.0 * unnormed_density.sum()

        avg_its = total_its / n_active if n_active > 0 else 0
        return loss, wd, ot_obj_values, avg_its

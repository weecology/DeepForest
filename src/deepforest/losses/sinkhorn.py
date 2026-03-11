"""PyTorch implementation of Sinkhorn-Knopp for entropic-regularized optimal
transport.

Based on ot.bregman.sinkhorn from the Python Optimal Transport library
(https://pythonot.github.io), rewritten to stay on-device without numpy round-trips.

Reference: M. Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal
Transport", NeurIPS 2013.
"""

import torch

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

    K = torch.exp(C / -reg)

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

    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return (P, log_dict) if log else P

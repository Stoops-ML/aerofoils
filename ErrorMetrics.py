import torch
import torch.nn.functional as F


def flatten_check(out, targ):
    """check that `out` and `targ` have the same number of elements and flatten them"""
    out, targ = out.contiguous().view(-1), targ.contiguous().view(-1)
    assert len(out) == len(targ), \
        f"Expected output and target to have the same number of elements but got {len(out)} and {len(targ)}."
    return out, targ


def root_mean_square(pred, targ):
    pred, targ = flatten_check(pred, targ)
    return torch.sqrt(F.mse_loss(pred, targ))


def R2_score(pred, targ):
    """R squared score"""
    pred, targ = flatten_check(pred, targ)
    u = torch.sum((targ - pred) ** 2)
    d = torch.sum((targ - targ.mean()) ** 2)
    return 1 - (u / d).item()


def top_losses(kwargs):
    return {k: v for k, v in sorted(kwargs.items(), key=lambda item: item[1], reverse=True)}

    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    # {0: 0, 2: 1, 1: 2, 4: 3, 3: 4}

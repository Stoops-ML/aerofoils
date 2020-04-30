import torch
import torch.nn.functional as F
import torch.nn as nn
import sys


def checks(func):
    def flatten_check(*args, **kwargs):
        """check that `out` and `targ` have the same number of elements and flatten them"""
        out, targ = args[0].contiguous().view(-1), args[1].contiguous().view(-1)
        assert len(out) == len(targ),\
            f"Expected output and target to have the same number of elements but got {len(out)} and {len(targ)}."
        return func(*args, **kwargs)
    return flatten_check


@checks
def root_mean_square(pred, targ):
    return torch.sqrt(F.mse_loss(pred, targ))


@checks
def R2_score(pred, targ):
    """R squared score"""
    u = torch.sum((targ - pred) ** 2)
    d = torch.sum((targ - targ.mean()) ** 2)
    return 1 - (u / d).item()


def top_losses(kwargs):
    return {k: v for k, v in sorted(kwargs.items(), key=lambda item: item[1], reverse=True)}


class MyLossFunc(nn.Module):
    def __init__(self):
        """loss function wrapper (instead of simply adding losses in for loop of epoch). The loss function is
        written this way so both the LRFinder() and the training loop can evaluate the losses in the same manner.
        This allows both methods to work in the same script."""
        super(MyLossFunc, self).__init__()
        self.loss_func1 = nn.SmoothL1Loss()
        self.loss_func2 = nn.SmoothL1Loss()

    def forward(self, *predictions_targets, **kwargs):  # matches forward of FindLR() (necessary)
        # predictions
        ClCd_prediction = predictions_targets[0][0]  # max ClCd at angle
        angle_prediction = predictions_targets[0][1]  # angle of max ClCd

        # targets
        ClCd_target = predictions_targets[1][:, :, 0]
        angle_target = predictions_targets[1][:, :, 1]

        # losses
        loss1 = self.loss_func1(ClCd_prediction, ClCd_target)
        loss2 = self.loss_func2(angle_prediction, angle_target)
        return loss1 + loss2

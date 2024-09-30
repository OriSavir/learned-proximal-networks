"""Learned proximal networks for input size of 28x28 (e.g. MNIST)."""

import torch
from torch import nn
from ..utils.norm_equiv import SortPool


def get_padding(kernel_size, dilation):
    return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2


class LPN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden,
        layers=1,
        beta=1,
        kernel_size=3,
        stride=1,
        dilation=1,
        alpha=0.0,
    ):
        super(LPN, self).__init__()

        self.hidden = hidden #Changes: No bias in layers, padding_mode is "reflect" instead of the default "zeros"
        self.lin = nn.ModuleList(
            [
                nn.Conv2d(
                    in_dim, hidden, 3, bias=False, stride=1, padding=1, dilation=1, padding_mode="reflect"
                ),  # 28
                nn.Conv2d(
                    hidden, hidden, 3, bias=False, stride=2, padding=1, dilation=1, padding_mode="reflect"
                ),  # 14
                nn.Conv2d(
                    hidden, hidden, 3, bias=False, stride=1, padding=1, dilation=1, padding_mode="reflect"
                ),  # 14
                nn.Conv2d(
                    hidden, hidden, 3, bias=False, stride=2, padding=1, dilation=1, padding_mode="reflect"
                ),  # 7
                nn.Linear(hidden * 7 * 7, 64, bias=False), #question: In NE LPN, we used AffineConv2d, which also had padding. Was that important / do the linear layers need it too?
                nn.Linear(64, 1, bias=False),
            ]
        )

        self.res = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1, dilation=1, bias=False, padding_mode="reflect"),  # 14
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1, dilation=1, bias=False, padding_mode="reflect"),  # 14
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1, dilation=1, bias=False, padding_mode="reflect"),  # 7
                nn.Linear(7 * 7 * in_dim, 64, bias=False),
            ]
        )

        self.act = SortPool() ## Interest: SortPool seems required as an activation function of NE property. However, it is not differentiable some places. Does this defeat the LPN theory?
        self.alpha = alpha

    def scalar(self, x):
        y = x.clone()
        y = self.act(self.lin[0](y))
        size = [28, 14, 14, 7]
        for core, res, sz in zip(self.lin[1:-2], self.res[:-1], size[:-1]):
            x_scaled = nn.functional.interpolate(x, (sz, sz), mode="bilinear")
            y = self.act(core(y) + res(x_scaled))

        y = y.reshape(y.shape[0], -1)
        x_scaled = nn.functional.interpolate(
            x, (size[-1], size[-1]), mode="bilinear"
        ).reshape(x.shape[0], -1)
        y = self.lin[-2](y) + self.res[-1](x_scaled)
        y = self.act(y)

        y = self.lin[-1](y)
        # return shape: (batch, 1)

        y = y**2 + self.alpha * x.pow(2).sum(dim=(1, 2, 3)).unsqueeze(1) ##changed here: square to make scale equivariant

        return y

    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def forward(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            #subtract mean of x here
            mean_x = x.mean(dim=(1,2,3), keepdim=True)
            x = x - mean_x
            x_ = x
            y = self.scalar(x_)
            grad = torch.autograd.grad(
                y.sum(), x_, retain_graph=True, create_graph=True
            )[0]
            #add mean of x back here
            grad = grad + mean_x

        return grad

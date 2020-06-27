import torch
import torch.nn as nn

from py_utils import kp, AELoss, _neg_loss, convolution, residual

class head(nn.Module):
    def __init__(self, dim):
        super(head, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

    def forward(self, x):

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        return conv2

class tl_head(head):
    def __init__(self, dim):
        super(tl_head, self).__init__(dim)

class br_head(head):
    def __init__(self, dim):
        super(br_head, self).__init__(dim)

class tr_head(head):
    def __init__(self, dim):
        super(tr_head, self).__init__(dim)

class bl_head(head):
    def __init__(self, dim):
        super(bl_head, self).__init__(dim)

def make_tl_layer(dim):
    return tl_head(dim)

def make_br_layer(dim):
    return br_head(dim)

def make_tr_layer(dim):
    return tr_head(dim)

def make_bl_layer(dim):
    return bl_head(dim)

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class model(kp):
    def __init__(self):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = 1

        super(model, self).__init__(
            n, 2, dims, modules, out_dim,
            make_tl_layer=make_tl_layer,
            make_br_layer=make_br_layer,
            make_tr_layer=make_tr_layer,
            make_bl_layer=make_bl_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

loss = AELoss(pull_weight=1e-1, push_weight=1e-1, focal_loss=_neg_loss)

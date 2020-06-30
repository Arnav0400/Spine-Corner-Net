import numpy as np
import torch
import torch.nn as nn

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss
from .kp_utils import make_tl_layer, make_br_layer, make_bl_layer, make_tr_layer, make_kp_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class kp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_tr_layer=make_tr_layer, make_bl_layer=make_bl_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(kp, self).__init__()

        self.nstack    = nstack
        self._decode   = _decode

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])
        self.tr_cnvs = nn.ModuleList([
            make_tr_layer(cnv_dim) for _ in range(nstack)
        ])
        self.bl_cnvs = nn.ModuleList([
            make_bl_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.tr_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.bl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.tr_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.bl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for tl_heat, br_heat, tr_heat, bl_heat in zip(self.tl_heats, self.br_heats, self.tr_heats, self.bl_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)
            tr_heat[-1].bias.data.fill_(-2.19)
            bl_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.tr_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.bl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image   = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]
        tr_inds = xs[3]
        bl_inds = xs[4]

        inter = self.pre(image)
        outs  = []
        
        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs, self.tr_cnvs, self.bl_cnvs,
            self.tl_heats, self.br_heats, self.tr_heats, self.bl_heats,
            self.tl_tags, self.br_tags, self.tr_tags, self.bl_tags,
            self.tl_regrs, self.br_regrs, self.tr_regrs, self.bl_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_, tr_cnv_, bl_cnv_ = layer[2:6]
            tl_heat_, br_heat_, tr_heat_, bl_heat_ = layer[6:10]
            tl_tag_, br_tag_, tr_tag_, bl_tag_   = layer[10:14]
            tl_regr_, br_regr_, tr_regr_, bl_regr_ = layer[14:18]

            kp  = kp_(inter)
            cnv = cnv_(kp)
            
            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)
            tr_cnv = tr_cnv_(cnv)
            bl_cnv = bl_cnv_(cnv)
            
            tl_heat, br_heat, tr_heat, bl_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), tr_heat_(tr_cnv), bl_heat_(bl_cnv)
            tl_tag,  br_tag, tr_tag, bl_tag = tl_tag_(tl_cnv),  br_tag_(br_cnv), tr_tag_(tr_cnv), bl_tag_(bl_cnv)
            tl_regr, br_regr, tr_regr, bl_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), tr_regr_(tr_cnv), bl_regr_(bl_cnv)

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            tr_tag  = _tranpose_and_gather_feat(tr_tag, tr_inds)
            bl_tag  = _tranpose_and_gather_feat(bl_tag, bl_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            tr_regr = _tranpose_and_gather_feat(tr_regr, tr_inds)
            bl_regr = _tranpose_and_gather_feat(bl_regr, bl_inds)

            outs += [tl_heat, br_heat, tr_heat, bl_heat, tl_tag, br_tag, tr_tag, bl_tag, tl_regr, br_regr, tr_regr, bl_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.tl_cnvs, self.br_cnvs, self.tr_cnvs, self.bl_cnvs,
            self.tl_heats, self.br_heats, self.tr_heats, self.bl_heats,
            self.tl_tags, self.br_tags, self.tr_tags, self.bl_tags,
            self.tl_regrs, self.br_regrs, self.tr_regrs, self.bl_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_, br_cnv_, tr_cnv_, bl_cnv_ = layer[2:6]
            tl_heat_, br_heat_, tr_heat_, bl_heat_ = layer[6:10]
            tl_tag_, br_tag_, tr_tag_, bl_tag_   = layer[10:14]
            tl_regr_, br_regr_, tr_regr_, bl_regr_ = layer[14:18]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                tr_cnv = tr_cnv_(cnv)
                bl_cnv = bl_cnv_(cnv)

                tl_heat, br_heat, tr_heat, bl_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), tr_heat_(tr_cnv), bl_heat_(bl_cnv)
                tl_tag,  br_tag, tr_tag, bl_tag = tl_tag_(tl_cnv),  br_tag_(br_cnv), tr_tag_(tr_cnv), bl_tag_(bl_cnv)
                tl_regr, br_regr, tr_regr, bl_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), tr_regr_(tr_cnv), bl_regr_(bl_cnv)

                outs += [tl_heat, br_heat, tr_heat, bl_heat, tl_tag, br_tag, tr_tag, bl_tag, tl_regr, br_regr, tr_regr, bl_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs[-12:]

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 12

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tr_heats = outs[2::stride]
        bl_heats = outs[3::stride]
        tl_tags  = outs[4::stride]
        br_tags  = outs[5::stride]
        tr_tags  = outs[6::stride]
        bl_tags  = outs[7::stride]
        tl_regrs = outs[8::stride]
        br_regrs = outs[9::stride]
        tr_regrs = outs[10::stride]
        bl_regrs = outs[11::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_tr_heat = targets[2]
        gt_bl_heat = targets[3]
        gt_mask    = targets[4]
        gt_tl_regr = targets[5]
        gt_br_regr = targets[6]
        gt_tr_regr = targets[7]
        gt_bl_regr = targets[8]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        tr_heats = [_sigmoid(t) for t in tr_heats]
        bl_heats = [_sigmoid(b) for b in bl_heats]
        
        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)
        focal_loss += self.focal_loss(tr_heats, gt_tr_heat)
        focal_loss += self.focal_loss(bl_heats, gt_bl_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag, tr_tag, bl_tag in zip(tl_tags, br_tags, tr_tags, bl_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, tr_tag, bl_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr, tr_regr, bl_regr in zip(tl_regrs, br_regrs, tr_regrs, bl_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
            regr_loss += self.regr_loss(tr_regr, gt_tr_regr, gt_mask)
            regr_loss += self.regr_loss(bl_regr, gt_bl_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss
        
        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)
        return loss.unsqueeze(0)

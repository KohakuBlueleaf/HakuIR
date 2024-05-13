# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
from functools import partial
from collections import OrderedDict, abc as container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from ..base import HakuIRModel


class WSConv2d(nn.Conv2d):
    """https://arxiv.org/abs/1903.10520"""

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        destination[f"{prefix}weight"] = self._get_weight()
        if self.bias is not None:
            destination[f"{prefix}bias"] = self.bias
        return destination

    def _get_weight(self, dtype=None):
        weight = self.weight
        eps = 1e-3 if (dtype or weight.dtype) == torch.float16 else 1e-5
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return normalized_weight

    def forward(self, x):
        return F.conv2d(
            x,
            self._get_weight(x.dtype),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SimpleChannelAttention(nn.Module):
    def __init__(self, c, expand=2, gate1=nn.Identity(), gate2=nn.Identity()):
        super().__init__()
        dw_channel = c * expand
        self.norm = nn.GroupNorm(min(c, 32), c, eps=1e-6)
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = WSConv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        self.gate1 = gate1
        self.gate2 = gate2

        # Simplified Channel Attention
        self.sca = nn.Linear(dw_channel // 2, dw_channel)
        torch.nn.init.constant_(self.sca.weight, 0)
        torch.nn.init.constant_(self.sca.bias, 0)

    def forward(self, x):
        # 1x1 + depth wise
        h = self.conv1(self.norm(h))
        h1, h2 = self.conv2(h).chunk(2, dim=1)
        h = self.gate1(h1) * self.gate2(h2)  # simple gate

        # SCA
        shift, scale = self.sca(h.mean(dim=[-1, -2], keepdim=True)).chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv3(h)
        return h


class GatedMLP(nn.Module):
    def __init__(self, c, expand=2, gate1=nn.Identity(), gate2=nn.Identity()):
        super().__init__()
        ffn_channel = expand * c

        self.norm = nn.GroupNorm(min(c, 32), c, eps=1e-6)
        self.conv1 = nn.Conv2d(c, ffn_channel, 1, 0)
        self.conv2 = nn.Conv2d(ffn_channel // 2, c, 1, 0)
        self.gate1 = gate1
        self.gate2 = gate2

    def forward(self, x):
        h1, h2 = self.conv1(self.norm(x)).chunk(2, dim=1)
        h = self.gate1(h1) * self.gate2(h2)
        output = self.conv2(h)
        return output


class KohakuBlock(nn.Module):
    def __init__(self, c, dw_expand=2, mlp_expand=2, drop_out_rate=0.0):
        super().__init__()
        self.sca = SimpleChannelAttention(
            c, expand=dw_expand, gate1=nn.Identity(), gate2=nn.Identity()
        )
        self.mlp = GatedMLP(
            c, expand=mlp_expand, gate1=nn.Identity(), gate2=nn.Identity()
        )

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        h = self.sca(x)
        h = self.dropout1(h)
        y = x + h * self.beta

        h = self.mlp(y)
        h = self.dropout2(h)
        return y + h * self.gamma


class KohakuNet(HakuIRModel):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[KohakuBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 3, 2, 1))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[KohakuBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[KohakuBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

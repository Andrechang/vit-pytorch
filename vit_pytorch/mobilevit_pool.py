#from: https://github.com/lucidrains/vit-pytorch/blob/1bae5d3cc58448f05d1252be306bbf48d9c5fede/vit_pytorch/mobile_vit.py

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Reduce
from .mobile_vit import MV2Block, conv_nxn_bn, conv_1x1_bn, depthconv_nxn_bn, FeedForward

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        '''b (ph pw) (h w) d: batch, patch, imgsz, channel
        b p h n d: batch, patch, head, imgsz, channel'''
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out), attn


class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.att = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.maxp = nn.MaxPool1d(2, stride=2)
        self.upp = nn.Upsample(scale_factor=2, mode='linear')
        self.pool = ''

    def forward(self, x):
        y = self.norm1(x)
        y, a = self.att(y)
        z = y + x
        b, _, _, _ = z.shape
        if self.pool == 'down':
            z = rearrange(z, 'b p n H -> (b p) H n')# because there isn't generic 1d maxpool
            z = self.maxp(z)
            z = rearrange(z, '(b p) H n -> b p n H', b=b)
        elif self.pool == 'up':
            z = rearrange(z, 'b p n H -> (b p) H n')  # because there isn't generic 1d maxpool
            z = self.upp(z)
            z = rearrange(z, '(b p) H n -> b p n H', b=b)

        y = self.norm2(z)
        o = self.ff(y) + z
        return o, a

class MobileViTBlock_v3(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., channel_out=0, ret_att=False):
        super().__init__()
        # print('MobileViTBlock_v3', dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout, channel_out, ret_att)
        self.ph, self.pw = patch_size

        self.conv1 = depthconv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.ret_att = ret_att
        depth = 2 if depth < 2 else depth
        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(Transformer(dim, 4, 8, mlp_dim, dropout))

        self.transformer[0].pool = 'down' #first transf
        self.transformer[-1].pool = 'up' #last transf

        if channel_out != channel and channel_out > 0:
            self.conv3 = conv_1x1_bn(channel, channel_out)
            self.conv4 = conv_1x1_bn(2 * dim, channel_out)
        else:
            self.conv3 = None
            self.conv4 = conv_1x1_bn(2 * dim, channel)

    def forward(self, x):
        y = x.clone()
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # Global representations
        _, _, h, w = x.shape
        z = x.clone() #local rep
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        a = []
        for transf in self.transformer:
            x, attn = transf(x)
            a.append(attn)

        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = torch.cat((x, z), 1)
        x = self.conv4(x)
        if self.conv3:
            y = self.conv3(y)
        if self.ret_att:
            return x + y, a
        else:
            return x + y

class MobileViT_pool(nn.Module):

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3),
        conv1=None,
        with_head=False,
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        mbvit_module = MobileViTBlock_v3

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels
        if conv1 is not None:
            self.conv1 = conv1
        else:
            self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([]) # resnet like backbone
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            mbvit_module(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            mbvit_module(dims[1], depths[1], channels[7],
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            mbvit_module(dims[2], depths[2], channels[9],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))
        self.with_head = with_head
        if with_head:
            self.to_logits = nn.Sequential(
                conv_1x1_bn(channels[-2], last_dim),
                Reduce('b c h w -> b c', 'mean'),
                nn.Linear(channels[-1], num_classes, bias=False)
            )

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)
        if self.with_head:
            x = self.to_logits(x)
        return x

def mobilevit_xxs_pool(num_classes):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT_pool((256, 256), dims, channels, num_classes=num_classes, expansion=4, with_head=True)

if __name__ == "__main__":
    import time
    from thop import profile
    model = mobilevit_xxs(10)
    input_ = torch.randn((1, 3, 256, 256))
    # layers = []
    # layers.append(MV2Block(256, 128, 2))
    # layers.append(MobileViTBlock(96, 2, 256, 3, (2, 2), int(96 * 2), channel_out=128, stride=1))
    # layers.append(MobileViTBlock_v3(96, 2, 256, 3, (2, 2), int(96 * 2)))
    # model = nn.Sequential(*layers)
    model_out = model(input_)
    macs, params = profile(model, inputs=(input_,)); print("encoder: macs, params: ", macs/(10**9), params)
    model = model.to('cuda')
    input_ = input_.to('cuda')
    start = time.time()
    # for i in range(1000):
    #     model_out = model(input_)
    # m = (time.time() - start) / 1000
    # print("% s seconds" % (m))
    print(model_out.shape)

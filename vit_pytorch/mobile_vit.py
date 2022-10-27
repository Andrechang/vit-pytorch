#from: https://github.com/lucidrains/vit-pytorch/blob/1bae5d3cc58448f05d1252be306bbf48d9c5fede/vit_pytorch/mobile_vit.py

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Reduce

# helpers

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def depthconv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernal_size, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
        nn.Conv2d(inp, oup, kernel_size=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

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

class FC_attn(nn.Module):
    def __init__(self, dim, dim_head=64, dropout=0.):
        super().__init__()
        self.to_v = nn.Linear(dim, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim_head, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, a):
        v = self.to_v(x)
        out = torch.matmul(a, v)
        return self.to_out(out)

class PTransformer(nn.Module):
    def __init__(self, dim, depth, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                FC_attn(dim, dim_head, dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x, al):
        """
        al: List of tensor
        """
        print(len(al), al[0].shape)
        for i, (norm, attn, ff) in enumerate(self.layers):
            y = norm(x)
            y = attn(y, al[i])
            y = y + x
            x = ff(y) + x
        return x

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.att = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        y = self.norm1(x)
        y, a = self.att(y)
        z = y + x
        y = self.norm2(z)
        o = self.ff(y) + z
        return o, a

class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        #expansion: hidden_dim expansion
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        # print('MV2Block', inp, oup, stride, expansion)
        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., channel_out=0, stride=1):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(Transformer(dim, 4, 8, mlp_dim, dropout))

        self.conv3 = conv_1x1_bn(dim, channel)
        if channel_out:
            self.conv4 = conv_nxn_bn(2 * channel, channel_out, kernel_size, stride=stride)
        else:
            self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                      ph=self.ph, pw=self.pw)
        for transf in self.transformer:
            x, _ = transf(x)

        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)

        return x

class MobileViTBlock_v3(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., channel_out=0, ret_att=False):
        super().__init__()
        # print('MobileViTBlock_v3', dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout, channel_out, ret_att)
        self.ph, self.pw = patch_size

        self.conv1 = depthconv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.ret_att = ret_att

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(Transformer(dim, 4, 8, mlp_dim, dropout))

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

class MobileTransViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., channel_out=0, scale=2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = depthconv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        dim_head = 64
        self.transformer = PTransformer(dim, depth, dim_head, mlp_dim, dropout)
        if channel_out != channel and channel_out > 0:
            self.conv3 = conv_1x1_bn(channel, channel_out)
            self.conv4 = conv_1x1_bn(2 * dim, channel_out)
        else:
            self.conv3 = None
            self.conv4 = conv_1x1_bn(2 * dim, channel)

        self.up = nn.Upsample(None, scale, 'nearest')

    def forward(self, x, attn):
        y = x.clone()
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # Global representations
        _, _, h, w = x.shape
        z = x.clone() #local rep
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                      ph=self.ph, pw=self.pw)
        x = self.transformer(x, attn)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = torch.cat((x, z), 1)
        x = self.conv4(x)
        if self.conv3:
            y = self.conv3(y)
        z = x + y
        z = self.up(z)
        return z

class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

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

def mobilevit_xxs(num_classes):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=num_classes, expansion=4, with_head=True)

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
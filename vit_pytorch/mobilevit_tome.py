import torch
import torch.nn as nn
import math
from typing import Callable, Tuple

from tome.merge import merge_source, merge_wavg
from tome.utils import parse_r
from mobile_vit import Attention, Transformer, MobileViT, mobilevit_xxs, MobileViTBlock_v3
from einops import rearrange

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :] # (b, hw, c) split half for hw dim
        scores = a @ b.transpose(-1, -2) #calc matching tokens
        print("scores: ", scores.shape)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        print("edge_idx: ", edge_idx.shape)
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, p, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, p, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, p, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, p, r, c), src, reduce=mode)
        print("unm_idx: ", unm_idx.shape)
        print("dst: ", dst.shape)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

class ToMeBlock(Transformer):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def forward(self, x):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None

        y = self.norm1(x)
        y, a, metric = self.att(y, attn_size)
        z = y + x

        r = self._tome_info["r"]#.pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(merge, z, self._tome_info["source"])
            z, self._tome_info["size"] = merge_wavg(merge, z, self._tome_info["size"])

        y = self.norm2(z)
        o = self.ff(y) + z
        return o, a

class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """
    def forward(self, x, size: torch.Tensor = None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply proportional attention
        # if size is not None:
        #     dots = dots + size.log()[:, None, None, :, 0]
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out), attn, k.mean(2)

class ToMeVisionTransformer(MobileViT):
    """
    Modifications:
     - Initialize r, token size, and token sources.
    """

    def forward(self, *args, **kwdargs) -> torch.Tensor:
        self._tome_info["r"] = self.r #parse_r(len(self.trunk), self.r)
        print(self._tome_info["r"], len(self.trunk))
        self._tome_info["size"] = None
        self._tome_info["source"] = None

        return super().forward(*args, **kwdargs)


def apply_patch(model, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    # model.__class__ = ToMeVisionTransformer
    model.r = 16
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Transformer):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention

if __name__ == "__main__":
    import time
    from thop import profile

    # model = mobilevit_xxs(10)
    # apply_patch(model)
    # print(model)
    # input_ = torch.randn((1, 3, 256, 256))
    input_ = torch.randn((1, 256, 16, 16))
    layers = []
    layers.append(MobileViTBlock_v3(96, 2, 256, 3, (2, 2), int(96 * 2)))
    model = nn.Sequential(*layers)
    apply_patch(model)

    model_out = model(input_)
    macs, params = profile(model, inputs=(input_,)); print("encoder: macs, params: ", macs / (10 ** 9), params)
    model = model.to('cuda')
    input_ = input_.to('cuda')
    start = time.time()
    for i in range(1000):
        model_out = model(input_)
    m = (time.time() - start) / 1000
    print("% s seconds" % (m))
    print(model_out.shape)
from torch import nn
import torch
from .MLP import MLP as FeedForward
from .PreNorm import PreNorm
from .SimilarAttention import SimilarAttention as Attention
from einops.layers.torch import Rearrange
from .RotateSimplifyLayer import RoSiLayer as ExtraLayer
# from MLP import MLP as FeedForward
# from PreNorm import PreNorm
# from SimilarAttention import SimilarAttention as Attention

class SimATransformer(nn.Module):
    def __init__(self, dim, depth, image_size, patch_size, mlp_dim, heads=8, dropout=0., RE_tag=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        patches_num = (image_size // patch_size) ** 2 + 1
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, patches_num=patches_num, dim_head=mlp_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        if RE_tag is True:
            self.lin1 = nn.Linear(dim, patch_size*patch_size)
            self.rear1 = Rearrange('b c (h p) -> b c h p', p=patch_size)
            self.extr = ExtraLayer(patch_size)
            self.rear2 = Rearrange('b c h p -> b c (h p)')
            self.lin2 = nn.Linear(patch_size*patch_size, dim)
        self.RE_tag = RE_tag

    def forward(self, x):
        if self.RE_tag is True:
            x = self.lin1(x)
            x = self.rear1(x)
            x = self.extr(x) + x
            x = self.rear2(x)
            x = self.lin2(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

if __name__ == '__main__':
    data = torch.rand(10, 50, 128)
    va = SimATransformer(dim=128, depth=1, image_size=224, patch_size=32, mlp_dim=64)
    out = va(data)
    print(out.shape)
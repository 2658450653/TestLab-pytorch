
from torch import nn
import torch
from vit_pytorch.cct import DropPath

from modules.RegionalCompressLayer import RegionalCompressLayer
from .MLP import MLP as FeedForward
from .PreNorm import PreNorm
from .SimilarAttention import SimilarAttention as Attention
#from .VeMAttention import VeAttention as Attention
from einops.layers.torch import Rearrange
from .RotateSimplifyLayer import RoSiLayer as ExtraLayer

# from MLP import MLP as FeedForward
# from PreNorm import PreNorm
# from SimilarAttention import SimilarAttention as Attention

class ConBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0)):
        super().__init__()
        self.r1 = Rearrange("(a e) b c -> a e b c", e=1)
        self.con = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.r2 = Rearrange("a e b c -> (a e) b c")

    def forward(self, x):
        x = self.r2(self.con(self.r1(x)))
        return x

class ECT_ST_Transformer(nn.Module):
    def __init__(self, dim, depth, image_size, patch_size, mlp_dim, heads=8, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        patches_num = (image_size // patch_size) ** 2 + 1
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, patches_num=patches_num, dim_head=mlp_dim, dropout=dropout)),
                #PreNorm(dim, Attention(dim, heads = heads, dim_head = mlp_dim, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ConBlock(in_channels=1, out_channels=1, kernel_size=(3, 1), padding=(1, 0)),
            ]))

    def forward(self, x):

        for attn, ff, conBlock in self.layers:
            x = attn(x) + x
            x = conBlock(x) + x
            x = ff(x) + x
        return x

if __name__ == '__main__':
    data = torch.rand(10, 50, 128)
    va = ECT_ST_Transformer(dim=128, depth=1, image_size=224, patch_size=32, mlp_dim=64)
    out = va(data)
    print(out.shape)
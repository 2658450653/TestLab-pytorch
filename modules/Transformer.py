from torch import nn

from .MLP import MLP as FeedForward
from .PreNorm import PreNorm
from .VeMAttention import VeAttention as Attention
from einops.layers.torch import Rearrange
from .RotateSimplifyLayer import RoSiLayer as ExtraLayer


class MyTransformer(nn.Module):
    def __init__(self, dim, depth, patch_size, heads, dim_head, mlp_dim, dropout=0., RE_tag=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        #print("RELayer is Ready!")
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        if RE_tag is True:
            self.lin1 = nn.Linear(dim, patch_size*patch_size//4)
            self.rear1 = Rearrange('b c (h p) -> b c h p', p=patch_size//2)
            self.extr = ExtraLayer(patch_size//2)
            self.rear2 = Rearrange('b c h p -> b c (h p)')
            self.lin2 = nn.Linear(patch_size*patch_size//4, dim)
        self.RE_tag = RE_tag

    def forward(self, x):
        if self.RE_tag is True:
            x = self.lin1(x)
            x = self.rear1(x)
            #print(x.shape)
            x = self.extr(x) + x
            #print(x.shape)
            x = self.rear2(x)
            #print(x.shape)
            x = self.lin2(x)
        
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        return x
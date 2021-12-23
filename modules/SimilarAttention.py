import torch
from einops import rearrange
from torch import nn
import math

class SimilarAttention(nn.Module):
    def __init__(self, dim, patches_num, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.attn_x = None  # save attention mask img
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkfv = nn.Linear(dim, inner_dim * 4, bias=False)
        self.scales = nn.Linear(dim_head, patches_num)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkfv = self.to_qkfv(x).chunk(4, dim=-1)
        q, k, f, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkfv)
        # (64, 8, 50, 50)
        mask = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # (64, 8, 128, 128)
        poss_value = torch.matmul(f.transpose(-1, -2), v)
        attn = self.attend(mask)
        poss_value = self.scales(poss_value)
        self.attn_x = attn
        out = torch.matmul(attn, poss_value.transpose(-1, -2))
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def AttensionMask(self, x):
        return self.attn_x

if __name__ == '__main__':
    data = torch.rand(10, 50, 128)
    va = SimilarAttention(patches_num=50, dim=128)
    out = va(data)
    print(out.shape)
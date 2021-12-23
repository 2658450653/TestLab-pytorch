import torch
from einops import rearrange
from torch import nn


class VeAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.attn_x = None #save attention mask img
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        self.attn_x = attn

        out = torch.matmul(attn, v)
        out = torch.matmul(out.transpose(-1, -2), attn)
        out = out.transpose(-1, -2)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def AttensionMask(self, x):
        return self.attn_x




if __name__ == '__main__':
    data = torch.rand(10, 50, 128)
    va = VeAttention(dim=128)
    out = va(data)
    print(out.shape)


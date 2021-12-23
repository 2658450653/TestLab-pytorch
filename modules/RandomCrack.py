import math

import torch
from torch import nn
from modules.RotateSimplifyLayer import RoSiLayer
from einops.layers.torch import Rearrange


class RandomCrack(nn.Module):
    def __init__(self, image_size, extra_type="crack", dropout=0.):
        super().__init__()
        self.image_size = image_size
        self.search = nn.Parameter(torch.randn(image_size, image_size, dtype=torch.float32) + 1)
        self.relu = nn.GELU()
        self.Dropout = nn.Dropout(dropout)
        if extra_type == "strength":
            self.extra_type = extra_type
        elif extra_type == "crack" :
            self.extra_type = extra_type
        elif extra_type == "rs":
            self.extra_type = extra_type
            self.rsLayer = RoSiLayer(image_size)
    def forward(self, x):
        if x.shape[-1] != x.shape[-2]:
            raise TypeError(f"the same width and height wanna be input, but get {x.shape[-1]} and {x.shape[-2]}.")
        #self.search = nn.Parameter(self.sm(self.search).to(torch.float32))
        x = self.Dropout(x)
        x = torch.matmul(self.search.transpose(-1, -2).float(), x.transpose(-1, -2).float()) / math.sqrt(self.image_size)
        x = torch.matmul(x.transpose(-1, -2).float(), self.search.float()) / math.sqrt(self.image_size)
        if self.extra_type == "crack":
            x = x + self.relu(x)
        elif self.extra_type == "strength":
            x = self.relu(x)
        elif self.extra_type == "rs":
            x = self.rsLayer(x) + x
        return x

if __name__ == '__main__':
    data = torch.arange(0, 4, dtype=torch.float32).reshape(1, 2, 2)
    data = data
    #x = torch.rand(20, 20)
    #print(torch.matmul(data, x))
    #print(data)
    rs = RandomCrack(2)
    out = rs(data)
    print(rs)
    print(out)
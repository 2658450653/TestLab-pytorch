import math

import torch
from torch import nn
from einops.layers.torch import Rearrange


class RoSiLayer(nn.Module):
    def __init__(self, image_size, dropout=0.):
        super().__init__()
        self.image_size = image_size
        img_area = image_size*image_size
        self.flat = Rearrange('b c h p -> b c (h p)')
        self.unflat = Rearrange('b c (h p) -> b c h p', h=image_size)
        self.search1 = nn.Linear(img_area, img_area*4)
        self.search2 = nn.Linear(img_area*4, img_area)
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.shape[-1] != x.shape[-2]:
            raise TypeError(f"the same width and height wanna be input, but get {x.shape[-1]} and {x.shape[-2]}.")
        #self.search = nn.Parameter(self.sm(self.search).to(torch.float32))
        x = self.flat(x)
        s = self.search1(x)
        s = self.search2(s)
        s = self.unflat(s)
        x = self.unflat(x)
        x = self.Dropout(x)
        x = torch.matmul(s.transpose(-1, -2).float(), x.transpose(-1, -2).float()) / math.sqrt(self.image_size)
        x = torch.matmul(x.transpose(-1, -2).float(), s.float()) / math.sqrt(self.image_size)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    data = torch.rand(3, 2, 4, 4, dtype=torch.float32)
    data = data
    #x = torch.rand(20, 20)
    #print(torch.matmul(data, x))
    #print(data)
    rs = RoSiLayer(4)
    out = rs(data)
    print(rs)
    print(out)
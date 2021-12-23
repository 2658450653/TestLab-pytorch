import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange

class RegionalCompressLayer(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.A = nn.Parameter(torch.randn(size=(out_size, input_size)))
        self.E = nn.Parameter(1 + torch.rand(size=(out_size, input_size)))
        self.bias = nn.Parameter(torch.zeros(input_size))

    def forward(self, x):
        x = x.unsqueeze(-2)
        x_x = x * x
        #print("x_x: ", x_x.shape)
        ab_x = (2 * self.A + self.E) * x
        #print("ab_x: ", ab_x.shape)
        ab = (self.A + self.E) * self.A
        #print("ab: ", ab.shape)
        ex = - x_x + ab_x -ab
        #print("ex: ", ex.shape)
        p = torch.log(math.sqrt(math.e - 1) + torch.exp(ex))
        #print("p: ", p.shape)
        out = x * p
        #print("out: ", out.shape)
        out = out.sum(-1)
        return out


if __name__ == "__main__":
    x = torch.arange(0, 36).reshape(2, 3, 6)
    y = torch.arange(0, 12).reshape(2, 1, 6)
    z = x * y
    print(z.shape)
    rc = RegionalCompressLayer(6, 10)
    out = rc(x)
    print(out.shape)


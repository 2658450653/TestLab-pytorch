# coding=utf-8
import torch
import torch.nn as nn

from modules.CBAM import ResBlock_CBAM
from modules.Conv import Conv


class CBAMTransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.att = ResBlock_CBAM(in_places=c2, places=c2//4)
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        return self.att(x)

if __name__ == '__main__':
    data = torch.rand(1, 256, 32, 32)
    f = CBAMTransformerBlock(256, 64)
    out = f(data)
    print(out.shape)
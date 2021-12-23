import torch
import torch.nn as nn


class OverlapCon2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size):
        super(OverlapCon2d, self).__init__()
        self.kernal_size = kernal_size
        self.con2d = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernal_size,
                               stride=kernal_size)

    def forward(self, x):
        _, c, w, h = x.shape
        assert w % self.kernal_size == 0 or h % self.kernal_size == 0
        x = self.con2d(x)
        x = torch.sum(x, dim=(-1, -2))
        x = x.unsqueeze(2)
        return x

data = torch.rand(1, 3, 224, 224)
oc = OverlapCon2d(in_channel=3, out_channel=16, kernal_size=32)
out = oc(data)
print(out.shape)

import torch
import torch.nn.functional as F
from torch import nn

from models.ShuffleNetv2 import BN_Conv2d, DSampling, BasicUnit


class ShuffleNetV2Backbone(nn.Module):
    """ShuffleNet-v2"""

    _defaults = {
        "sets": {0.5, 1, 1.5, 2},
        "units": [3, 7],
        "chnl_sets": {0.5: [16, 32, 64],
                      1: [32, 64, 256],
                      1.5: [48, 96, 320],
                      2: [64, 128, 152]}
    }

    def __init__(self, scale, is_se=False, is_res=False) -> object:
        super(ShuffleNetV2Backbone, self).__init__()
        self.__dict__.update(self._defaults)
        assert (scale in self.sets)
        self.is_se = is_se
        self.is_res = is_res
        self.chnls = self.chnl_sets[scale]

        # make layers
        self.conv1 = BN_Conv2d(3, self.chnls[0], 3, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stage2 = self.__make_stage(self.chnls[0], self.chnls[1], self.units[0])
        self.stage3 = self.__make_stage(self.chnls[1], self.chnls[2], self.units[1])

    def __make_stage(self, in_chnls, out_chnls, units):
        layers = [DSampling(in_chnls),
                  BasicUnit(2 * in_chnls, out_chnls, self.is_se, self.is_res)]
        for _ in range(units-1):
            layers.append(BasicUnit(out_chnls, out_chnls, self.is_se, self.is_res))
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = self.body(x)
        out = self.conv1(x)
        out = self.maxpool(out)
        #print(out.shape)
        out = self.stage2(out)
        #print(out.shape)
        out = self.stage3(out)
        #print(out.shape)
        return out

if __name__ == '__main__':
    model = ShuffleNetV2Backbone(0.5)
    data = torch.rand(1, 3, 512, 512)
    out = model(data)
    print(out.shape)
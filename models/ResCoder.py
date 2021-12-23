import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import gc


class ResCoder(nn.Module):
    def __init__(self, input_channel, out_channel, image_size=224, dropout=0):
        super(ResCoder, self).__init__()
        coder_dim_list = [8, 16, 32, 64, out_channel]

        coder_dim_list = [input_channel, *coder_dim_list]

        temp = []
        self.relu = nn.LeakyReLU()
        # 1*1 卷积
        i = 1
        self.con_11 = nn.Conv2d(coder_dim_list[i - 1], coder_dim_list[i], kernel_size=1)
        self.bn11 = nn.BatchNorm2d(coder_dim_list[i])
        # 3*3 卷积
        self.con_12 = nn.Conv2d(coder_dim_list[i], coder_dim_list[i], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(coder_dim_list[i])
        self.drop1 = nn.Dropout(dropout)

        # 1*1 卷积
        i = 2
        self.con_21 = nn.Conv2d(coder_dim_list[i - 1], coder_dim_list[i], kernel_size=1)
        self.bn21 = nn.BatchNorm2d(coder_dim_list[i])
        # 3*3 卷积
        self.con_22 = nn.Conv2d(coder_dim_list[i], coder_dim_list[i], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(coder_dim_list[i])
        self.drop2 = nn.Dropout(dropout)

        # 1*1 卷积
        i = 3
        self.con_31 = nn.Conv2d(coder_dim_list[i - 1], coder_dim_list[i], kernel_size=1)
        self.bn31 = nn.BatchNorm2d(coder_dim_list[i])
        # 3*3 卷积
        self.con_32 = nn.Conv2d(coder_dim_list[i], coder_dim_list[i], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(coder_dim_list[i])
        self.drop3 = nn.Dropout(dropout)

        # 1*1 卷积
        i = 4
        self.con_41 = nn.Conv2d(coder_dim_list[i - 1], coder_dim_list[i], kernel_size=1)
        self.bn41 = nn.BatchNorm2d(coder_dim_list[i])
        # 3*3 卷积
        self.con_42 = nn.Conv2d(coder_dim_list[i], coder_dim_list[i], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(coder_dim_list[i])
        self.drop4 = nn.Dropout(dropout)

        # 1*1 卷积
        i = 5
        self.con_51 = nn.Conv2d(coder_dim_list[i - 1], coder_dim_list[i], kernel_size=1)
        self.bn51 = nn.BatchNorm2d(coder_dim_list[i])
        # 3*3 卷积
        self.con_52 = nn.Conv2d(coder_dim_list[i], coder_dim_list[i], kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(coder_dim_list[i])
        self.drop5 = nn.Dropout(dropout)

        self.modules1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=2, dilation=2,
                      stride=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(input_channel)
        )

        self.modules2 = nn.Sequential(
            nn.Linear(image_size, 1),
            Rearrange("a b c d -> a b (c d)"),
            nn.Linear(image_size, 1),
            Rearrange("a b c -> a (b c)"),
        )

    def forward(self, x):
        x = self.modules1(x)
        x = self.con_11(x)
        x = self.bn11(x)
        x = self.con_12(x) + x
        x = self.bn12(x)
        x = self.relu(x)
        x = self.drop1(x)
        #print(x.shape)
        x = self.con_21(x)
        x = self.bn21(x)
        x = self.con_22(x) + x
        x = self.bn22(x)
        x = self.relu(x)
        x = self.drop2(x)
        #print(x.shape)
        x = self.con_31(x)
        x = self.bn31(x)
        x = self.con_32(x) + x
        x = self.bn32(x)
        x = self.relu(x)
        x = self.drop3(x)
        #print(x.shape)
        x = self.con_41(x)
        x = self.bn41(x)
        x = self.con_42(x) + x
        x = self.bn42(x)
        x = self.relu(x)
        x = self.drop4(x)
        #print(x.shape)
        x = self.con_51(x)
        x = self.bn51(x)
        x = self.con_52(x) + x
        x = self.bn52(x)
        x = self.relu(x)
        x = self.drop5(x)
        #print(x.shape)
        x = self.modules2(x)
        gc.collect()
        #print(x.shape)
        return x

if __name__ == "__main__":
    data = torch.rand(1, 3, 224, 224).cuda()
    model = ResCoder(input_channel=3, image_size=224, out_channel=128).to('cuda')
    out = model(data)
    print(out.shape)

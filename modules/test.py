import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from RegionalCompressLayer import RegionalCompressLayer as RCLayer
from utils.SeedTorch import seed_everything

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
seed_everything(32)
x = torch.unsqueeze(torch.linspace(-10, 10, 400), dim=1)  # 将一维数据构建成二维矩阵
y = 0.2 * torch.rand(x.size()) + torch.sin(x) # 添加噪点
x, y = Variable(x), Variable(y)  # 将x,y中的数据封装成variable类型


class Net(torch.nn.Module):  # 自定义网络继承了Pytorch的module
    """
    n_feature:特征量
    n_hidden:隐藏层的神经元数量
    n_output:输出量
    """

    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()  # 官方输出一下符类init
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  # 构建隐藏层
        self.rc = RCLayer(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 构建预测层即输出层

    def forward(self, x):  # 前向传播
        x1 = F.relu(self.hidden1(x))  # 使用relu激活函数
        #x1 = self.rc(x1)
        return self.predict(x1)


net = Net(1, 150, 1)  # 创建net对象
print(net)  # 输出net

plt.ion()  # 打开交互流可连续显示图像
plt.show()

optimizer = torch.optim.Adam(net.parameters(), lr=0.05)  # 选择SGD梯度下降算法
loss_func = torch.nn.MSELoss()  # 使用均方误差


mode='cosineAnnWarm'
if mode=='cosineAnn':
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
elif mode=='cosineAnnWarm':
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)


for t in range(500):  # 训练500次
    prediction = net(x)  # 获得prediction
    loss = loss_func(prediction, y)  # 获得误差

    optimizer.zero_grad()  # 清除上一步的梯度（因为使用的位SGD固每次下降只需要当前的梯度）
    loss.backward()  # 反向计算梯度
    optimizer.step()  # 下降一步

    if t % 15 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-')
        #plt.savefig(r'C:\Users\Y_ch\Desktop\torch_test\%d.png' % (t))
        plt.pause(0.1)
plt.ioff()  # 关闭交互流
plt.show()
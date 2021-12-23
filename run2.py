import gc
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from numpy import random
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import trange
import torch.nn.functional as F

# from models.ECT_ST_ViT import ECT_ST_ViT
from models.myCCViT import myCCViT
from models.ECT_ST_ViT import ECT_ST_ViT
from utils.SeedTorch import seed_everything
from modules.RegionalCompressLayer import RegionalCompressLayer as RCLayer

# 1.Globals Parameters
# 数据量越大，编码长度越长；
# 深度存在一些特定的值，但不会随着深度增加而持续提高精度；
# 注意力的数量较大程度取决于模型的架构以及数据分布
batch_size = 64
epochs = 100
lr = 0.001
gamma = 0.3
# 21 26 31 36 41 46 51
seed = 26
device = 'cuda'
image_size = 48
patch_size = 48
channels = 3
dropout = 0.
saveName = 'classifier/ResCoder'
# loss_func = "MSE"
loss_func = "CrossEntropy"
mode = 'cosineAnnWarm'
#mode = 'cosineAnn'

loss_list = []
acc_list = []
val_loss_list = []
val_acc_list = []
best_acc = 0
es = 0
max_es = 15
anchor = 1

seed_everything(seed)

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        # transforms.RandomRotation(36),
        # transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform1 = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class CIFAR10(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True, alpha=1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.alpha = alpha
        self.train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                           'data_batch_4', 'data_batch_5']
        self.test_list = ['test_batch']

        self.data = []
        self.targets = []

        if (self.train):
            choose_list = self.train_list
        else:
            choose_list = self.test_list

        for file_name in choose_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')  # 解析py2存储的文件
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # vstack它是垂直（按照行顺序）的把数组给堆叠起来
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC，维度转换
        self.dataLen = len(self.data)
        self.data_processed = []
        self.init_data()
        self.clean()

    def clean(self):
        del self.data
        del self.root
        # del self.transform
        # del self.target_transform
        del self.targets
        del self.test_list
        del self.train_list
        gc.collect()

    def init_data(self):
        for i in range(int(len(self.data)*self.alpha)):
            self.data_processed.append(self.getitem(i))

    def __getitem__(self, index):
        img, target = self.data_processed[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def getitem(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        return img, target

    def __len__(self):
        return int(self.dataLen*self.alpha)

# 0.813 32,1,32,32
# 0.823 32,2,32,32
# 0.825 32,2,32,64
# 0.822 32,2,32,128
# 0.818 64,2,32,32
# 0.811 64,2,32,64
# 0.817 32,4,32,32
# 0.794 32,6,32,32
model = ECT_ST_ViT(image_size=image_size,
                   patch_size=patch_size,
                   num_classes=10,
                   dim=32,
                   depth=2,
                   heads=32,
                   mlp_dim=64,
                   dropout=dropout
                   ).to(device)

# import torch
# from thop import profile
#
# input = torch.randn(1, 3, 112, 112)
# flops, params = profile(model, (input,))
# print('flops: ', flops, 'params: ', params)


# model = myCCViT(scale=0.5,
#                 depth_rcb=3,
#                 image_size=image_size,
#                 num_classes=10,
#                 dim=32,
#                 depth_tr=2,
#                 heads=8,
#                 mlp_dim=32,
#                 dropout=dropout).to(device)

# model = models.resnet50(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# fc_inputs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(fc_inputs, 10),
#     # nn.ReLU(),
#     # nn.Linear(400, 200),
#     # nn.ReLU(),
#     # nn.Linear(200, 10), # 0.6399489
# )
# model = model.to('cuda:0')

# print("model parameters init.")
# for m in model.modules():
#     if isinstance(m, (nn.Conv2d, nn.Linear)):
#         nn.init.kaiming_normal_(m.weight, mode='fan_in')
# print("model parameters inited.")

# import torch
# from torchvision.models import resnet18
# from thop import profile
# input = torch.randn(1, 3, 224, 224)
# model.to('cpu')
# flops, params = profile(model, inputs=(input, ))
# print('flops:{}'.format(flops))
# print('params:{}'.format(params))

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__ == '__main__':
    root = 'data/cifar-10-batches-py'
    train_set = CIFAR10(root, transform=transform)
    test_set = CIFAR10(root, transform=transform1, train=False)
    batchSZ = batch_size

    train_loader = DataLoader(train_set, batch_size=batchSZ, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batchSZ, shuffle=True, num_workers=0, pin_memory=True)

    valid_data = test_set
    valid_loader = test_loader

    # loss function

    loss_strategy = {"MSE": nn.MSELoss(), "CrossEntropy": nn.CrossEntropyLoss()}
    criterion = loss_strategy[loss_func]
    #criterion = LabelSmoothLoss(0.1)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler_strategy = {'cosineAnn': CosineAnnealingLR(optimizer, T_max=5, eta_min=0),
                          'cosineAnnWarm': CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
                          'reduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=12,
                                                                 verbose=False, threshold=0.00001, threshold_mode='rel',
                                                                 eps=1e-08)}
    scheduler = scheduler_strategy[mode]

    epoch_loss = 0
    epoch_accuracy = 0

    print("train data and loader: ", len(train_set), len(train_loader))
    print("valid data and loader", len(valid_data), len(valid_loader))
    print("test data and loader", len(test_set), len(test_loader))
    d = {"Counter": f"{0}/{max_es}", "loss": 0, "acc": 0, "val_loss": 0, "val_acc": 0, "val_acc_best": 0}
    import copy

    temp_model = copy.deepcopy(model)
    with trange(epochs) as t:
        for i in t:
            epoch_loss = 0
            epoch_accuracy = 0

            for data, label in train_loader:
                data = data.to(device)
                # print("data:", data.shape)
                label = label.to(device)
                # print("label:", label[0])
                output = model(data)
                output.to(device)
                # print("out:", output[0])
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc_strategy = {"CrossEntropy": lambda output: (output.argmax(dim=1) == label).float().mean(),
                                "MSE": lambda output: (
                                        torch.max(output, 1)[1] == torch.max(label, 1)[1]).float().mean()}
                acc = acc_strategy[loss_func](output)
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)
                gc.collect()
            scheduler.step()
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in test_loader:
                    data = data.to(device)
                    label = label.to(device)
                    val_output = model(data)
                    val_output.to(device)
                    val_loss = criterion(val_output, label)
                    val_acc = acc_strategy[loss_func](val_output)
                    epoch_val_accuracy += val_acc / len(test_loader)
                    epoch_val_loss += val_loss / len(test_loader)
            if epoch_val_accuracy > best_acc:
                best_acc = epoch_val_accuracy
                es = 0
                d["Counter"] = f"{es}/{max_es}"
                d["val_acc_best"] = best_acc.cpu().detach().numpy()
                del temp_model
                temp_model = copy.deepcopy(model)
                temp_model.to("cpu")
                # torch.save(model, saveName + f'{0}.pth')
            else:
                es += 1
                d["Counter"] = f"{es}/{max_es}"
                d["val_acc_best"] = best_acc.cpu().detach().numpy()
                # if es >= max_es:
                #     print("Early stopping with best_acc: ", best_acc)
                #     torch.save(temp_model, saveName + f'{d["val_acc_best"]}.pth')
                #     break
            d["loss"] = epoch_loss.cpu().detach().numpy()
            d["acc"] = epoch_accuracy.cpu().detach().numpy()
            d["val_loss"] = epoch_val_loss.cpu().detach().numpy()
            d["val_acc"] = epoch_val_accuracy.cpu().detach().numpy()
            gc.collect()
            t.set_postfix(d)

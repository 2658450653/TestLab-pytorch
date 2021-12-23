# 2021-12-23
CIFAR: 0.8883359

batch_size = 64
lr = 0.001
image_size = 48
patch_size = 48
dropout = 0

model = ECT_ST_ViT(image_size=image_size,
                   patch_size=patch_size,
                   num_classes=10,
                   dim=32,
                   depth=2,
                   heads=32,
                   mlp_dim=64,
                   dropout=dropout
                   ).to(device)
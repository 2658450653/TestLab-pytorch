import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from models.ViT import PreNorm, FeedForward
from modules.CBAM_transformer import CBAMTransformerBlock
from modules.ECT_ST_Transformer import ConBlock
from modules.RegionalCompressLayer import RegionalCompressLayer as RCLayer
from modules.ShuffleNetV2Backbone import ShuffleNetV2Backbone
from modules.SimilarAttention import SimilarAttention as Attention


class classifierBlock(nn.Module):
    def __init__(self, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, pool = 'cls', channels = 3,
                 dropout = 0., emb_dropout = 0.):
        super(classifierBlock, self).__init__()

        self.to_patch_embedding = nn.Sequential(
            # 将图片分割为b*h*w个三通道patch，b表示输入图像数量
            Rearrange('b c p1 p2 -> b c (p1 p2)', p1=patch_size, p2=patch_size),
            # 经过线性全连接后，维度变成(64, 49, 128)
            nn.Linear(patch_size*patch_size, dim),
        )
        # dim, depth, image_size, patch_size, mlp_dim, heads=8, dropout
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, patches_num=channels+1, dim_head=mlp_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ConBlock(in_channels=1, out_channels=1, kernel_size=(3, 1), padding=(1, 0)),
            ]))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 224),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(224, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        #print("cat ", x.shape)
        for attn, ff, conBlock in self.layers:
            x = attn(x) + x
            x = conBlock(x) + x
            x = ff(x) + x
        #print(x.shape)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        # 最终进行分类映射
        return self.mlp_head(x)


class myCCViT(nn.Module):
    _defaults = {
        "sets": {0.5, 1, 1.5, 2},
        "chnl_sets": {0.5: [16, 32, 64],
                      1: [32, 64, 256],
                      1.5: [48, 96, 320],
                      2: [64, 128, 152]}
    }
    def __init__(self, scale, depth_rcb, image_size, num_classes, dim, depth_tr, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(myCCViT, self).__init__()
        self.__dict__.update(self._defaults)
        self.backbone = ShuffleNetV2Backbone(scale, is_se=False, is_res=True)
        in_ch = self.chnl_sets[scale][-1]
        self.ResCBAMBlock = [CBAMTransformerBlock(in_ch, in_ch) for _ in range(depth_rcb)]
        self.ResCBAMBlock = nn.Sequential(*self.ResCBAMBlock)
        self.tr = classifierBlock(image_size // 16,
                                  num_classes,
                                  dim,
                                  depth_tr,
                                  heads,
                                  mlp_dim,
                                  pool,
                                  in_ch,
                                  dropout,
                                  emb_dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.ResCBAMBlock(x)
        #print("ResCBAMBlock", x.shape)
        x = self.tr(x)
        return x

if __name__ == '__main__':
    data = torch.rand(1, 3, 512, 512)
    f = myCCViT(scale=0.5,
                depth_rcb=4,
                image_size=512,
                num_classes=10,
                dim=128,
                depth_tr=4,
                heads=8,
                mlp_dim=128)
    out = f(data)
    print(out.shape)
import torch
import torchvision.models as models
from einops.layers.torch import Rearrange
from torch import nn

from modules.ECT_ST_Transformer import ECT_ST_Transformer as Transformer
from modules.RegionalCompressLayer import RegionalCompressLayer as RCLayer

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ECT_ST_ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                pool='cls', channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.coder = models.resnet50(pretrained=True)
        for param in self.coder.parameters():
            param.requires_grad = True
        fc_inputs = self.coder.fc.in_features
        self.coder.fc = nn.Sequential(
            nn.Linear(fc_inputs, dim*(num_patches+1)),
        )

        self.restore = nn.Sequential(
            Rearrange('b (a s) -> b a s', s=dim)
        )

        self.to_patch_embedding = self.coder

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # 64, 50, 128
        self.transformer = Transformer(dim=dim,
                                       depth=depth,
                                       heads=heads,
                                       image_size=image_size,
                                       patch_size=patch_size,
                                       mlp_dim=mlp_dim,
                                       dropout=dropout,
                                       )

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

        self.conBlock = nn.Sequential(
            Rearrange("(a e) b c -> a e b c", e=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding=(0, 1)),
            Rearrange("a e b c -> (a e) b c"),
        )

    def forward(self, img):
        x = img

        x = self.to_patch_embedding(x)
        x = self.restore(x)

        b, n, _ = x.shape

        x = self.transformer(x)
        x = self.conBlock(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

if __name__ == '__main__':
    import torch
    from thop import profile
    data = torch.rand(1, 3, 48, 48)
    model = ECT_ST_ViT(image_size=48, patch_size=48, num_classes=3, dim=32, depth=2, heads=32, mlp_dim=64)
    flops, params = profile(model, (data,))
    print('flops: ', flops, 'params: ', params)

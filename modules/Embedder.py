import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Embedder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, channels=3):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim-1),
        )
        self.location = torch.arange(0, num_patches, dtype=torch.float32).reshape(1, num_patches, -1) / num_patches
        self.location = nn.Parameter(self.location, requires_grad=False)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.consult = nn.Linear(dim, 1)
        self.con_sig = nn.Sigmoid()

    def forward(self, img):
        x = img
        x = self.to_patch_embedding(x)
        x = torch.cat((self.location, x), dim=-1)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return x


if __name__ == '__main__':
    data = torch.rand(1, 3, 224, 224)
    embedder = Embedder(image_size=224, patch_size=32, dim=128, channels=3)
    out = embedder(data)
    print(out.shape)

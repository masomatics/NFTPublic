import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from src.model.mae_modeling_pretrain import PretrainVisionTransformerDecoder, PretrainVisionTransformerEncoder


class Coder(nn.Module):
    def __init__(self, img_size, img_channels, **kwargs):
        super().__init__()
        self.net = self.build(img_size, img_channels, **kwargs)
    
    def build(self, img_size, img_channels, **kwargs):
        pass

    def forward(self, input):
        return self.net(input)


class Conv(Coder):
    def build(self, img_size, img_channels, **kwargs):
        if img_size == 28:
            modules = [
                nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=4, stride=2, padding=1), 
                nn.LeakyReLU(), # 28 -> 14
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(), # 14 -> 7
                Rearrange('n c h w -> n (h w) c'),
            ]
        return nn.Sequential(*modules)

class Deconv(Coder):
    def build(self, img_size, img_channels, **kwargs):
        if img_size == 28:
            modules = [
                Rearrange('n (h w) c -> n c h w', h=7),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            ]
        return nn.Sequential(*modules)

# class Conv28(Coder):
#     def build(self, in_channels, **kwargs):
#         modules = [
#             nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1), 
#             nn.LeakyReLU(), # 28 -> 14
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(), # 14 -> 7
#             Rearrange('n c h w -> n (h w) c'),
#             # nn.Flatten(),
#             # nn.Linear(64 * 7 * 7, out_channels),
#         ]
#         return nn.Sequential(*modules)


class Conv64(Coder):
    def build(self, in_channels, out_channels, **kwargs):
        modules = [
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(), # 64 -> 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(), # 32 -> 16
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(), # 16 -> 8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, out_channels)
        ]
        return nn.Sequential(*modules)


# class Deconv28(Coder):
#     def build(self, out_channels, **kwargs):
#         modules = [
#             # nn.Linear(in_channels, 64 * 7 ** 2),
#             # nn.LeakyReLU(),
#             # nn.Unflatten(1, (64, 7, 7)),
#             Rearrange('n (h w) c -> n c h w', h=7),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         ]
#         return nn.Sequential(*modules)


class Deconv64(Coder):
    def build(self, in_channels, out_channels, **kwargs):
        modules = [
            nn.Linear(in_channels, 128 * 8 ** 2),
            nn.LeakyReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ]
        return nn.Sequential(*modules)


class Linear64(Coder):
    def build(self, in_channels, out_channels, **kwargs):
        modules = [
            Rearrange('b c h w -> b (c h w)'), 
            nn.Linear(64 * 64 * in_channels, out_channels)
        ]
        return nn.Sequential(*modules)


class Delinear64(Coder):
    def build(self, in_channels, out_channels, **kwargs):
        modules = [
            nn.Linear(in_channels, 64 * 64 * out_channels),
            Rearrange('b (c h w) -> b c h w', c=out_channels, h=64, w=64), 
            nn.Sigmoid()
        ]
        return nn.Sequential(*modules)


def ResDeconvBlock(in_channels, out_channels, img_size):
    return nn.Sequential(
        Residual(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU(),
            nn.LayerNorm([in_channels, img_size, img_size])
        )),
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
        nn.LayerNorm([out_channels, img_size * 2, img_size * 2])
        )


class ResDeconv64(Coder):
    def build(self, in_channels, out_channels, **kwargs):
        modules = [
            nn.Linear(in_channels, 128 * 8 ** 2),
            nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            ResDeconvBlock(128, 64, 8),
            ResDeconvBlock(64, 32, 16),
            ResDeconvBlock(32, 16, 32),
            nn.ConvTranspose2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ]
        return nn.Sequential(*modules)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixerBlock(width, kernel_size, num_patches):
    return nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(width, width, kernel_size, groups=width, padding="same"),
                    nn.GELU(),
                    nn.LayerNorm([width, num_patches, num_patches])
                    # LayerNorm(width, data_format='channel_first'),
                )),
                # nn.Conv2d(width, 2 * width, 1),
                # nn.GELU(),
                # nn.Conv2d(2 * width, width, 1),
                nn.Conv2d(width, width, 1),
                nn.GELU(),
                nn.LayerNorm([width, num_patches, num_patches])
                # LayerNorm(width, data_format='channel_first'),
        )


def ConvMixerBlock2(width, kernel_size, num_patches):
    return nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(width, width, kernel_size, groups=width, padding="same"),
                    nn.LayerNorm([width, num_patches, num_patches]),
                    nn.Conv2d(width, 4 * width, 1),
                    nn.GELU(), 
                    nn.Conv2d(4 * width, width, 1),
                )),
        )


class ConvMixer(Coder):
    def build(self, in_channels, out_channels, img_size, width, patch_size, depth, kernel_size, **args):
        num_patches = img_size // patch_size
        modules = [
            nn.Conv2d(in_channels, width, kernel_size=patch_size, stride=patch_size),
            # nn.GELU(),
            nn.LayerNorm([width, num_patches, num_patches]),
            # LayerNorm(width, data_format='channel_first'),
            *[ConvMixerBlock(width, kernel_size, num_patches) for i in range(depth)],
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(width * num_patches ** 2, out_channels)
            # Reduce('b c h w -> b c', 'mean'),
            # nn.Linear(width, out_channels)
        ]
        return nn.Sequential(*modules)

class ViTEncoder(Coder):
    def build(self, img_size, img_channels, **kwargs):
        kwargs['img_size'] = img_size
        kwargs['in_chans'] = img_channels
        return PretrainVisionTransformerEncoder(**kwargs)

# class ViTEncoder(Coder):
#     def build(self, in_channels, out_channels, molding, **kwargs):
#         embed_dim = kwargs['embed_dim']
#         num_patches = (kwargs['img_size'] // kwargs['patch_size']) ** 2
#         # action_dim = out_channels // embed_dim
#         if molding == 'straight':
#             adapter = nn.Sequential(
#                 Rearrange('b n c -> b (n c)'),
#                 nn.Linear(num_patches * embed_dim, out_channels),
#             )
#         elif molding == 'embed':
#             adapter = nn.Sequential(
#                 nn.Linear(embed_dim, out_channels),
#                 Rearrange('b n o -> b (n o)')
#             )
#         elif molding == 'patch':
#             adapter = nn.Sequential(
#                 Rearrange('b n c -> b c n'),
#                 nn.Linear(num_patches, out_channels),
#                 Rearrange('b c o -> b (c o)')
#             )
#         elif molding == 'lowrank':
#             ratio = 4
#             edim_out = embed_dim // ratio
#             np_out = num_patches // ratio
#             adapter = nn.Sequential(
#                 nn.Linear(embed_dim, edim_out),
#                 Rearrange('b n c -> b c n'),
#                 nn.Linear(num_patches, np_out),
#                 nn.GELU(),
#                 nn.LayerNorm([edim_out, np_out]),
#                 Rearrange('b c n -> b (c n)'),
#                 nn.Linear(edim_out * np_out, out_channels)
#             )
#         kwargs['in_chans'] = in_channels
#         return nn.Sequential(
#             PretrainVisionTransformerEncoder(**kwargs),
#             adapter
#         )

class ViTDecoder(Coder):
    def build(self, img_size, img_channels, **kwargs):
        psize = kwargs['patch_size']
        kwargs['num_classes'] = img_channels * psize ** 2
        return nn.Sequential(
            PretrainVisionTransformerDecoder(**kwargs),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      p1=psize, p2=psize, h=img_size // psize)
        )

# class ViTDecoder(Coder):
#     def build(self, in_channels, out_channels, img_size, molding, encoder_embed_dim, **kwargs):
#         embed_dim = kwargs['embed_dim']
#         # action_dim = in_channels // embed_dim
#         psize = kwargs['patch_size']
#         num_patches = (img_size // psize) ** 2
#         if molding == 'straight':
#             adapter = nn.Sequential(
#                 nn.Linear(in_channels, embed_dim * num_patches),
#                 Rearrange('b (n c) -> b n c', n=num_patches)
#             )
#         elif molding == 'embed':
#             adapter = nn.Sequential(
#                 Rearrange('b (n o) -> b n o', o=in_channels),
#                 nn.Linear(in_channels, embed_dim)
#             )
#         elif molding == 'patch':
#             adapter = nn.Sequential(
#                 Rearrange('b (c o) -> b c o', o=in_channels),
#                 nn.Linear(in_channels, num_patches),
#                 Rearrange('b c n -> b n c'),
#                 nn.Linear(encoder_embed_dim, embed_dim),
#             )
#         elif molding == 'lowrank':
#             ratio = 4
#             edim_in = embed_dim // ratio
#             np_in = num_patches // ratio
#             adapter = nn.Sequential(
#                 nn.Linear(in_channels, edim_in * np_in),
#                 Rearrange('b (c n) -> b c n', n=np_in),
#                 nn.GELU(),
#                 nn.LayerNorm([edim_in, np_in]),
#                 nn.Linear(np_in, num_patches),
#                 Rearrange('b c n -> b n c'),
#                 nn.Linear(edim_in, embed_dim),
#             )

#         kwargs['num_classes'] = out_channels * psize ** 2
#         return nn.Sequential(
#             adapter,
#             PretrainVisionTransformerDecoder(**kwargs),
#             Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
#                       p1=psize, p2=psize, h=img_size // psize)
#         )        


class Linear28(Coder):
    H = W = 28
    def build(self, in_channels, out_channels, **kwargs):
        self.out_channels = out_channels
        modules = [
            nn.Linear(in_channels, out_channels * self.H * self.W),
            nn.Sigmoid()
        ]
        return nn.Sequential(*modules)

    def forward(self, z):
        output = self.net(z)
        return output.view([-1, self.out_channels, self.H, self.W])


class Logit(Coder):
    def build(self, in_channels, out_channels, num_classes, **kwargs):
        return nn.Linear(in_channels, num_classes)
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum
from einops.layers.torch import Rearrange
from src.model.base_lightning_model import BaseLightningModel

class LitMSPAE(BaseLightningModel):
    def __init__(self, latent_dim, action_dim, num_patches, enc_embed_dim, dec_embed_dim, encoder, decoder):
        super().__init__()

        total_dim = latent_dim * action_dim
        self.pre_adapter = nn.Sequential( 
            Rearrange('b n c -> b (n c)'),
            nn.Linear(num_patches * enc_embed_dim, total_dim),
            Rearrange('b (h a) -> b h a', h=latent_dim),
        )
        self.post_adapter = nn.Sequential(
            Rearrange('b h a -> b (h a)'),
            nn.Linear(total_dim, dec_embed_dim * num_patches),
            Rearrange('b (n c) -> b n c', n=num_patches),
            nn.LayerNorm([num_patches, dec_embed_dim]),
        )
        self.encoder, self.decoder = encoder, decoder
        self.action_dim = action_dim
        self.U = None
        self.block_dim_list = []

    def enc(self, img):
        latent = self.encoder(img)
        latent = self.pre_adapter(latent)
        return latent

    def dec(self, latent):
        latent = self.post_adapter(latent)
        latent = self.decoder(latent)
        return latent

    # def trans(self, latent):
    #     '''
    #     Estimate M using the latents, and predict the latents of the next state. 
    #     Note: the prediction is only made for the last latent z[-1].
    #         latent: len_sequence x batch x latent_dim x action_dim
    #     '''
    #     M = self.estimate_M(latent)
    #     pred_latent = einsum(M, latent[-1], 'b a_in a_out, b h a_in -> b h a_out')

    def amplitudes(self, latent):
        if self.U is None:
            return latent.norm(dim=-1)
        latent_cob = einsum(self.U, latent, 'ain i, b h ain -> b h i')
        latent_cob = torch.Tensor(latent_cob)
        invariant_features = []
        for block in self.block_dim_list:
            inv = latent_cob[..., block].norm(dim=-1)
            invariant_features.append(inv)
        return torch.stack(invariant_features, dim=-1)


    def estimate_M(self, latent0, latent1):
        '''
        M = argmin_M || AM - B ||^2 where A = H0^T H0 and B = H0^T H1
        '''
        A = einsum(latent0, latent0, 'b h a1, b h a2 -> b a1 a2')
        B = einsum(latent0, latent1, 'b h a1, b h a2 -> b a1 a2')
        return torch.linalg.solve(A, B)

    def forward(self, input):
        '''
        Given image pairs xs = [x, g x], predict the next image g^2 x.
            input: (2 x batch) x C x H x W
            output: batch x C x H x W
        '''
        latent = self.enc(input)
        latent = rearrange(latent, '(seq b) ... -> seq b ...', seq=2)
        M = self.estimate_M(latent[0], latent[1])
        pred_latent = einsum(M, latent[-1], 'b a_in a_out, b h a_in -> b h a_out')
        output = self.dec(pred_latent)
        return output

    # def training_step(self, batch, batch_idx):
    #     loss = super().training_step(batch, batch_idx)
    #     self.log('loss/train', loss)
    #     return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split, (loss, x_triplet) = super().validation_step(batch, batch_idx, dataloader_idx)
        self.log(f'reconst_err/{split}', loss, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        if batch_idx == 0:
            self.save_image(x_triplet, split)

    def loss(self, inputs, train):
        target = inputs[-1]
        pair = rearrange(inputs[:-1], 'seq b ... -> (seq b) ...')
        target_pred = self(pair)
        loss = F.mse_loss(target_pred, target)
        if train:
            return loss
        
        input = inputs[-2]
        return loss, (input, target, target_pred)

    def train_loss(self, inputs, params, ys):
        return self.loss(inputs, train=True)

    def test_loss(self, inputs, params, ys):
        return self.loss(inputs, train=False)


if __name__ == '__main__':
    import pyrootutils
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git"],
        pythonpath=True,
        dotenv=True,
    )
    from src.model.network import Conv, Deconv

    batch = 16
    H = 28
    latent_dim, action_dim, num_patches = 20, 10, 49
    encoder = Conv(img_size=H, img_channels=3)
    decoder = Deconv(img_size=H, img_channels=3)
    enc_embed_dim, dec_embed_dim = 64, 64

    model = LitMSPAE(latent_dim, action_dim, num_patches, enc_embed_dim, dec_embed_dim, encoder, decoder)
    xs = torch.rand(3, batch, 3, H, H).unbind(0)
    xs = list(xs)
    print(model.train_loss(xs, None, None))
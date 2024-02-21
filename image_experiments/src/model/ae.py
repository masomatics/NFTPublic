from src.model.base_lightning_model import BaseLightningModel
import torch.nn.functional as F


class LitAutoEnc(BaseLightningModel):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def enc(self, x):
        z = self.encoder(x)
        return z

    def dec(self, z):
        x = self.decoder(z)
        return x

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log('loss/train', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split, loss = super().validation_step(batch, batch_idx, dataloader_idx)
        self.log(f'loss/{split}', loss, sync_dist=True, prog_bar=True, add_dataloader_idx=False)

    def train_loss(self, xs, params, ys):
        x = xs[0]
        x_reconst = self.dec(self.enc(x))
        return F.mse_loss(x_reconst, x)

    def test_loss(self, xs, params, ys):
        return self.train_loss(xs, params, ys)
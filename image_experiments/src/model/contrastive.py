from torch import nn
from src.model.base_lightning_model import BaseLightningModel
from src.util.infonce import info_nce

class LitContrastiveEnc(BaseLightningModel):
    def __init__(self, encoder, decoder, enc_total_dim, head_dim, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(enc_total_dim, head_dim)
        )
        self.temperature = temperature

    def enc(self, x):
        z = self.encoder(x)
        return z.flatten(start_dim=1)

    def proj(self, x):
        z = self.enc(x)
        z = self.head(z)
        return z

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log('infoNCE/train', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split, loss = super().validation_step(batch, batch_idx, dataloader_idx)
        self.log(f'infoNCE/{split}', loss, sync_dist=True, prog_bar=True, add_dataloader_idx=False)

    ### currently only applicable when n_views = 1
    def train_loss(self, xs, params, ys):
        (x1, x2) = xs
        z1 = self.proj(x1)
        z2 = self.proj(x2)
        return info_nce(z1, z2, temperature=self.temperature)

    def test_loss(self, xs, params, ys):
        return self.train_loss(xs, params, ys)
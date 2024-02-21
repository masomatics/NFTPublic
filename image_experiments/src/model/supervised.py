from src.model.base_lightning_model import BaseLightningModel
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

class LitSupervised(BaseLightningModel):
    def __init__(self, encoder, decoder, encoder_dim, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_dim, num_classes),
        )        
        self.metric = Accuracy(task='multiclass', num_classes=num_classes)


    def enc(self, x):
        z = self.encoder(x)
        return z

    def dec(self, z):
        x = self.decoder(z)
        return x

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log('CrossEntropy/train', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split, loss = super().validation_step(batch, batch_idx, dataloader_idx)
        self.log(f'acc/{split}', loss, sync_dist=True, prog_bar=True, add_dataloader_idx=False)

    def train_loss(self, xs, params, y):
        x = xs[0]
        y_pred = self.dec(self.enc(x))
        return F.cross_entropy(y_pred, y)

    def test_loss(self, xs, params, y):
        x = xs[0]
        y_pred = self.dec(self.enc(x))
        logit = F.softmax(y_pred, dim=-1)
        return self.metric(logit, y)
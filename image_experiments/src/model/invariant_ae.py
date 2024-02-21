import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from src.model.base_lightning_model import BaseLightningModel
from src.model.ae import LitAutoEnc


class LitInvariantAE(LitAutoEnc):

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split, loss = BaseLightningModel.validation_step(self, batch, batch_idx, dataloader_idx)
        self.log(f'reconst_err/{split}', loss, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        if batch_idx == 0:
            xs, _, _ = batch
            x = xs[0]
            x_reconst = self.dec(self.enc(x))
            x_reconst = torchvision.utils.make_grid(x_reconst, normalize=True)
            self.logger.experiment.add_image(f'prediction/{split}', x_reconst, self.current_epoch)

    ### this works only when n_view = 1 and action = SO2
    def train_loss(self, xs, params, ys):
        trans_angle = 360 * params[-1] / (2 * torch.pi)
        x = xs[0]
        x = self.dec(self.enc(x))
        x = torch.stack(list(map(TF.rotate, x.unbind(dim=0), trans_angle.tolist())), dim=0)
        return F.mse_loss(x, xs[-1])
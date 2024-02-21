import torch
import torchvision
import lightning as pl
from hydra.utils import instantiate

class BaseLightningModel(pl.LightningModule):
    def on_train_start(self):
        if self.current_epoch == 0:
            self.logger.log_hyperparams(self.hparams, {"reconst_err/test": 1, "angle_err/test": 1})

    def training_step(self, batch, batch_idx):
        xs, params, ys = batch
        if not isinstance(xs, list):
            xs = list(xs.unbind(dim=1))
            params = list(params.unbind(dim=1))
        loss = self.train_loss(xs, params, ys)
        self.log('loss/train', loss)
        return loss

    def train_loss(self, xs, params, ys):
        pass
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        SPLITS = ['test', 'ood', 'ood2']
        split = SPLITS[dataloader_idx]
        
        xs, params, ys = batch
        if not isinstance(xs, list):
            xs = list(xs.unbind(dim=1))
            params = list(params.unbind(dim=1))

        return split, self.test_loss(xs, params, ys)

    def save_image(self, x_triplet, split, num_img=8):
        x_ref, x_target, x_reconst = x_triplet
        x_img = torch.cat([x_ref[:num_img], 
                           x_reconst[:num_img], 
                           x_target[:num_img]], dim=0)
        x_img = torchvision.utils.make_grid(x_img, normalize=True, nrow=num_img)
        self.logger.experiment.add_image(f'prediction/{split}', x_img, self.current_epoch)

    def test_loss(self, xs, params, ys):
        pass

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        return optimizer
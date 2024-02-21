import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, einsum
from einops.layers.torch import Rearrange
import src.model.action
from src.model.base_lightning_model import BaseLightningModel
from src.util.eval import mse2psnr

def check_params(params):
    if torch.is_tensor(params):
        return [params]
    return params


class EncoderAdapter(nn.Module):
    def __init__(self, num_patches, embed_dim, d_a, d_m):
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            Rearrange('b n c -> b c n'),
            nn.Linear(num_patches, num_patches // 4),
            nn.GELU(),
            nn.LayerNorm([embed_dim // 4, num_patches // 4]),
            Rearrange('b c n -> b (c n)'),
            nn.Linear(embed_dim * num_patches // 16, d_a * d_m),
            Rearrange('b (m a) -> b m a', m=d_m),
        )
    
    def forward(self, encoder_output):
        return self.net(encoder_output)


class ActionList():
    def __init__(self, action_list):
        self._actions = []
        self.total_dim = 0
        self.inverse_func = []

        if not isinstance(action_list, list):
            action_list = [action_list]

        for param_index, act_cfg in enumerate(action_list):
            num_basis, group_name, include_identity \
                = [act_cfg.pop(key) for key in ['num_basis', 'group', 'include_identity']]
            for basis_id in range(num_basis + 1):
                if basis_id == 0 and not include_identity:
                    continue
                act_cfg['freq'] = basis_id
                act = getattr(src.model.action, group_name)(**act_cfg)
                self._actions.append([act, param_index])
                self.total_dim += act.action_dim
            self.inverse_func.append(act.invert_param)

    def __len__(self):
        return len(self._actions)

    def __iter__(self):
        yield from self._actions

    def __getitem__(self, index):
        return self._actions[index]


class LitEquivariantAE(BaseLightningModel):
    def __init__(self, latent_dim, num_patches, enc_embed_dim, dec_embed_dim,
                 action_list, encoder, decoder, latent_pooling='average', adapter='lowrank', rank_ratio=4):
        super().__init__()
        self.num_actions = len(action_list)
        self.actions = action_list
        self.inverse_func = action_list.inverse_func
        self.action_dim = action_list.total_dim
        self.latent_pooling = getattr(self, f'{latent_pooling}_pooling')

        self.encoder = encoder
        self.decoder = decoder

        total_dim = latent_dim * self.action_dim
        if adapter == 'straight':
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
        elif adapter == 'lowrank':
            np_in = np_out = num_patches // rank_ratio
            edim_in = dec_embed_dim // rank_ratio
            edim_out = enc_embed_dim // rank_ratio
            self.pre_adapter = nn.Sequential(
                nn.Linear(enc_embed_dim, edim_out),
                Rearrange('b n c -> b c n'),
                nn.Linear(num_patches, np_out),
                nn.GELU(),
                nn.LayerNorm([edim_out, np_out]),
                Rearrange('b c n -> b (c n)'),
                nn.Linear(edim_out * np_out, total_dim),
                Rearrange('b (h a) -> b h a', h=latent_dim),
            )
            self.post_adapter = nn.Sequential(
                Rearrange('b h a -> b (h a)'),
                nn.Linear(total_dim, edim_in * np_in),
                Rearrange('b (c n) -> b c n', n=np_in),
                nn.GELU(),
                nn.LayerNorm([edim_in, np_in]),
                nn.Linear(np_in, num_patches),
                Rearrange('b c n -> b n c'),
                nn.Linear(edim_in, dec_embed_dim),
            )

    def enc(self, img, logit=False):
        latent = self.encoder(img)
        # latent = rearrange(latent, 'b (h a) -> b h a', a=self.action_dim)
        latent = self.pre_adapter(latent)
        return latent

    def dec(self, latent):
        latent = self.post_adapter(latent)
        # latent = rearrange(latent, 'b h a -> b (h a)')
        latent = self.decoder(latent)
        return latent

    def trans(self, z, params):
        params = check_params(params)
        zout = torch.zeros_like(z)
        for z_i, zout_i, (action, param_index) in zip(self.to_chunks(z), 
                                                      self.to_chunks(zout), 
                                                      self.actions):
            zout_i[:] = action.trans(z_i, params[param_index])
            # rep = action.rep(params[param_index]).to(dtype=z.dtype)
            # zout_i[:] = einsum(z_i, rep, 'b h a_in, b a_in a_out -> b h a_out')
            # # zout_i[:] = einsum(rep, z_i, 'b a_out a_in, b h a_in -> b h a_out')
        return zout

    def invert_param(self, params):
        params = check_params(params)
        params_out = []
        for i, param in enumerate(params):
            params_out.append(self.inverse_func[i](param))
        return params_out

    def average_pooling(self, xs, params):
        z_pred = 0
        for x, param in zip(xs, params):
            z_pred_i = self.trans(self.enc(x), self.invert_param(param))
            z_pred += z_pred_i / len(xs)
        return z_pred

    def to_chunks(self, latent, action_dim=-1):
        start_dim = 0
        neurons = []
        for action, _ in self.actions:
            neurons.append(latent.narrow(dim=action_dim, start=start_dim, length=action.action_dim))
            start_dim += action.action_dim
        return neurons

    def softmax_pooling(self, xs, params):
        zs = []
        logits = []
        for x, param in zip(xs, params):
            z, logit = self.enc(x, logit=True)
            z = self.trans(z, self.invert_param(param))
            zs.append(z)
            logits.append(logit)
        zs = torch.stack(zs, dim=-1)
        logits = torch.stack(logits, dim=-1)

        z_pred = torch.zeros_like(z)
        for i, (zs_i, z_pred_i) in enumerate(zip(self.to_chunks(zs, action_dim=-2), 
                                                 self.to_chunks(z_pred))):
            logit_i = logits[..., i, :]
            z_pred_i[:] = einsum(F.softmax(logit_i, dim=-1), zs_i, 'b h view, b h a view -> b h a')
        return z_pred

    def sim(self, z1, z2):
        return F.cosine_similarity(z1, z2)

    def normalize(self, z):
        zout = torch.ones_like(z)
        for z_i, zout_i in zip(self.to_chunks(z), self.to_chunks(zout)):
            zout_i[:] = F.normalize(z_i, dim=-1)
        return zout

    def amplitudes(self, z):
        amplitudes = []
        for z_i in self.to_chunks(z):
            amplitudes.append(z_i.norm(dim=-1))
        return torch.stack(amplitudes, dim=-1)

    # def coef_and_phase(self, z):
    #     coef = self.amplitudes(z)
    #     phase = 

    def angle_variance(self, z):
        z = self.normalize(z)
        variances = []
        for z_i in self.to_chunks(z):
            if z_i.shape[-1] == 1:
                continue
            var = 1 - z_i.mean(dim=-2).norm(dim=-1)
            variances.append(var)
        return torch.stack(variances, dim=-1)

    def forward(self, input, params=None):
        z = self.enc(input)
        if params is not None:
            z = self.trans(z, params)
        output = self.dec(z)
        return output

    def identify(self, z1, z2, target_freq=None):
        '''
        estimate the group parameters
        '''
        z1 = self.normalize(z1)
        z2 = self.normalize(z2)
        est_params = []
        for z1_i, z2_i, (action, _) in zip(self.to_chunks(z1), 
                                           self.to_chunks(z2), 
                                           self.actions):
            # if action.freq == 1:
            #     theta = action.identify(z1_i, z2_i)
            #     est_params.append(theta)
            if target_freq is not None and action.freq != target_freq:
                continue
            theta = action.identify(z1_i, z2_i)
            est_params.append(theta)
        if len(est_params) == 1:
            est_params = est_params[0]
        return est_params

    def param_loss(self, param1, param2):
        action, _ = self.actions[0]
        return action.param_loss(param1, param2)

    #################################
    # functions used for pytorch lightning
    #################################

    def on_train_start(self):
        if self.current_epoch == 0:
            self.logger.log_hyperparams(self.hparams, {"reconst_err/test": 1, 
                                                       "reconst_err/ood": 1, 
                                                       "reconst_err/ood2": 1, 
                                                       "angle_err/test": 1})

    # def training_step(self, batch, batch_idx):
    #     loss = super().training_step(batch, batch_idx)
    #     self.log('loss/train', loss)
    #     return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split, (loss, angle_loss, x_triplet) = super().validation_step(batch, batch_idx, dataloader_idx)
        self.log(f'reconst_err/{split}', loss, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f'PSNR/{split}', mse2psnr(loss), sync_dist=True, add_dataloader_idx=False)
        self.log(f'angle_err/{split}', angle_loss, sync_dist=True, add_dataloader_idx=False)
        if batch_idx == 0:
            self.save_image(x_triplet, split)
            
    def train_loss(self, xs, params, ys):
        loss_config = self.hparams.loss
        x_last, param_last = xs.pop(), params.pop()
        z_last = self.enc(x_last)
        z_avg = self.latent_pooling(xs, params)
        z_last_pred = self.trans(z_avg, param_last)

        loss = 0
        if loss_config.pred:
            x_last_pred = self.dec(z_last_pred)
            loss += F.mse_loss(x_last_pred, x_last)
            if len(xs) == 1:
                z_first_pred = self.trans(z_last, self.invert_param(param_last))
                z_first_pred = self.trans(z_first_pred, params[0])
                x_first_reconst = self.dec(z_first_pred)
                loss += F.mse_loss(xs[0], x_first_reconst)
        
        if loss_config.pred_l1:
            x_last_pred = self.dec(z_last_pred)
            loss += F.l1_loss(x_last_pred, x_last)

        if loss_config.reconst:
            x_last_reconst = self.dec(z_last)
            loss += F.mse_loss(x_last_reconst, x_last)

        if loss_config.alignl2:
            loss += loss_config.equiv_coef * F.mse_loss(z_last, z_last_pred)

        if loss_config.nalignl2:
            z_last = self.normalize(z_last)
            z_last_pred = self.normalize(z_last_pred)
            loss += loss_config.equiv_coef * F.mse_loss(z_last, z_last_pred)

        if loss_config.angle_variance:
            loss += self.angle_variance(z_last_pred).mean()
            
        return loss

    def test_loss(self, xs, params, ys):
        x, param = xs.pop(), params.pop()
        z_avg = self.latent_pooling(xs, params)
        z_pred = self.trans(z_avg, param)
        x_pred = self.dec(z_pred)
        loss = F.mse_loss(x_pred, x)

        params_est = self.identify(z_avg, self.enc(x), target_freq=1)
        angle_loss = self.param_loss(param, params_est)
        return loss, angle_loss, (xs[0], x, x_pred)

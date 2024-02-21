from einops.layers.torch import Rearrange
from models.vit import ViTDecoder, ViTEncoder

class SeqAELSTSQ_vit(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            alignment=False,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            change_of_basis=False,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            detachM=0,
            vit_args={},
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive

        total_dim = dim_a*dim_m
        self.vit_args = vit_args
        self.num_patches = int((self.vit_args['img_size']/self.vit_args['patch_size'])**2)

        self.pre_adapter = nn.Sequential( 
            Rearrange('b n c -> b (n c)'),
            nn.Linear(self.num_patches * self.vit_args['embed_dim'], total_dim),
            Rearrange('b (h a) -> b h a', h=self.dim_m),
        )
        self.post_adapter = nn.Sequential(
            Rearrange('b h a -> b (h a)'),
            nn.Linear(total_dim, self.vit_args['embed_dim'] * self.num_patches),
            Rearrange('b (n c) -> b n c', n=self.num_patches),
            nn.LayerNorm([self.num_patches, self.vit_args['embed_dim']]),
        )

        print(vit_args)

        self.enc = ViTEncoder(**self.vit_args)
        self.dec = ViTDecoder(**self.vit_args)
        self.dynamics_model = LinearTensorDynamicsLSTSQ(alignment=alignment)
        self.detachM = detachM
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)


    def _encode_base(self, xs, enc):
        #batch, time, ch, h, w
        shape = xs.shape
        x = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))
        H = enc(x)  #batch*time x embed_dimen xnumquery
        H = self.pre_adapter(H)   #batch*time x ds x da 
        H = torch.reshape(
            H, (shape[0], shape[1], *H.shape[1:]))   #batch  x time x ds x da 
        return H

    def encode(self, xs):
        H = self._encode_base(xs, self.enc)
        # batch*time x flatten_dimen
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(self.change_of_basis,
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])

        return H

    def decode(self, H):
        #expects inputs of shape   #batch  x time x ds x da 
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(torch.linalg.inv(self.change_of_basis),
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        n, t = H.shape[:2]

        #post adapter takes ds da input shape and turn it into dq de ouput
        H = rearrange(H, 'n t d_s d_a -> (n t) d_s d_a')

        H = self.post_adapter(H)   #(n t) dq de
        x_next_preds = self.dec(H)      #(n t) imageshape
        x_next_preds = torch.reshape(
            x_next_preds, (n, t, *x_next_preds.shape[1:]))  
        return x_next_preds
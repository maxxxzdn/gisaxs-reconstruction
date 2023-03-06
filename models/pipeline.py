import torch
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from .cvae import cVAE


class Pipeline(pl.LightningModule):
    def __init__(self, n_layers: int, n_transforms: int, hidden_dim:int,
                 cvae_params: dict, lr: float = 1e-3, step_lr: int = 10):
        """
        Main model: cVAE + NFs
        Args:
            n_layers (int): number of layers in material
            n_transforms (int): number of transforms in the NF
            hidden_dim (int): hidden dimension of the NF
            cvae_params (dict): parameters of the cVAE
            lr (float): learning rate
            step_lr (int): step size for learning rate scheduler
        see cvae.py for more details on cVAE parameters
        """
        super().__init__()
        self.cvae = cVAE(**cvae_params)
        self.flow = self.construct_flow(n_layers, n_transforms, hidden_dim, self.cvae.context_dim)
        self.lr = lr
        self.step_lr = step_lr
        self.n_params = n_layers*6
        self.encoder_dim = self.cvae.latent_dim
                
    def construct_flow(self, n_layers, n_transforms, hidden_dim, context_dim) -> Flow:
        """
        Constructs a NF: a composite transform of a number of masked affine autoregressive transforms
        and a number of reverse permutations, followed by a conditional diagonal normal distribution.
        Args:
            n_layers (int): number of layers in material
            n_transforms (int): number of transforms in the NF
            hidden_dim (int): hidden dimension of the NF
            context_dim (int): context dimension of the NF
        """
        base_dist = ConditionalDiagonalNormal(
            shape=(n_layers*6,), 
            context_encoder=torch.nn.Linear(context_dim, 2*6*n_layers)
        )

        transforms = []
        for _ in range(n_transforms):
            transforms.append(ReversePermutation(features=6*n_layers))
            transforms.append(MaskedAffineAutoregressiveTransform(features=6*n_layers, 
                                                                  hidden_features=hidden_dim, 
                                                                  context_features=context_dim))

        transform = CompositeTransform(transforms)
        return Flow(transform, base_dist)
              
    def loss(self, inputs, context):
        """
        Loss function for the NFs (negative log-likelihood).
        Args:
            inputs (torch.Tensor): input tensor
            context (torch.Tensor): context tensor
        """
        return -self.flow.log_prob(inputs=inputs, context=context)
             
    def __call__(self, pattern, profile, params):
        """
        Forward pass of the model.
        Computes the loss, the elbo, the NF loss, the KL divergence, and the reconstruction.
        Args:
            pattern (torch.Tensor): 2D GISAXS pattern
            profile (torch.Tensor) in-plane profile
            params (torch.Tensor): parameters of a sample
        """
        bs = params.shape[0]
        z, mu, std = self.cvae.encode(pattern, profile)
        inputs = self.cvae.get_context(profile, z)
        ############### NFs ###################
        params = params.repeat(self.cvae.n_samples, 1)
        nf_loss = self.loss(params, inputs).reshape(-1, self.cvae.n_samples).mean(1)
        ############### CVAEs ###################
        x_hat = self.cvae.decoder(inputs)
        pattern = pattern.reshape(bs, 1, 128, 16)
        # reconstruction loss
        recon_loss = self.cvae.gaussian_likelihood(x_hat, self.cvae.log_scale, pattern)
        # kl
        kl = self.cvae.kl_divergence(
            z.reshape(self.cvae.n_samples, bs, self.cvae.latent_dim), 
            mu.unsqueeze(0), 
            std.unsqueeze(0)
        ).mean(0)
        # elbo
        assert kl.shape == recon_loss.shape == nf_loss.shape
        elbo = kl - recon_loss
        loss = (elbo + nf_loss).mean()
        return loss, elbo.mean(), nf_loss.mean(), kl.mean(), (x_hat, profile, params)
            
    def training_step(self, batch, batch_idx):
        """
        Returns the loss for a single batch.
        """
        loss, elbo, nf_loss, kl, (x_hat, profile, params) = self(*batch)
        lp = self.cvae_log_prob(x_hat, profile, params).mean()
        self.log_dict({"kl": kl.item(), "nf_loss": nf_loss.item(), "elbo": elbo.item(), "lp": lp.item()}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.
        """
        loss, elbo, nf_loss, kl, (x_hat, profile, params) = self(*batch)
        lp = self.cvae_log_prob(x_hat, profile, params).mean()
        self.log_dict({"val_kl": kl.item(), "val_loss": loss.item(), "val_nf_loss": nf_loss.item(), 
                       "elbo": elbo.item(), "val_lp": lp.item()}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Defines the optimizer and the learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_lr, gamma=0.1)
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler}}
    
    def cvae_log_prob(self, pattern, profile, params, std=0.1):
        """
        Computes the log-probability of a sample under the cVAE.
        Args:
            pattern (torch.Tensor): 2D GISAXS pattern
            profile (torch.Tensor) in-plane profile
            params (torch.Tensor): parameters of a sample
            std (float): standard deviation of the latent space (higher covers more of the latent space)
        """
        with torch.no_grad():
            z = std*torch.randn(params.shape[0], 32, self.cvae.latent_dim).to(pattern.device).reshape(-1, self.cvae.latent_dim)
            inputs = self.cvae.get_context(profile, z, 32)
            params = params.repeat(32, 1)
            return -self.loss(params, inputs)
            
    def inference(self, pattern, n_samples):
        """
        Sample parameters from the NFs.
        Args:
            pattern (torch.Tensor): 2D GISAXS pattern
            n_samples (int): number of samples
        """
        with torch.no_grad():
            context = self.encoder(pattern.view(-1, 128*16))
            params = self.flow.sample(n_samples, context)
            return params.cpu().detach()
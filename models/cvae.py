import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class View(nn.Module):
    """
    Reshapes a tensor to a given shape.
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class cVAE(pl.LightningModule):
    def __init__(self, latent_dim: int, context_dim: int, hidden_dim_enc: int, hidden_dim_dec: int,
                 n_samples:int = 1, lr: float = 1e-3, step_lr: int = 10, drop_prob = 0.):
        """
        Conditional Variational Autoencoder.
        Args:
            latent_dim (int): latent dimension
            context_dim (int): context dimension
            hidden_dim_enc (int): hidden dimension of the encoder
            hidden_dim_dec (int): hidden dimension of the decoder
            n_samples (int): number of samples to draw from the latent space
            lr (float): learning rate
            step_lr (int): step size for learning rate scheduler
            drop_prob (float): dropout probability
        """
        super().__init__()
        self.n_samples = n_samples
        self.lr = lr
        self.step_lr = step_lr
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        # encoder X -> z
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim_enc, kernel_size=3, padding=1, stride=(2,1)), # 128x16 => 64x16
            nn.BatchNorm2d(hidden_dim_enc),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),
            nn.Conv2d(hidden_dim_enc, 4*hidden_dim_enc, kernel_size=3, padding=1, stride=(2,1)), # 64x16 => 32x16
            nn.BatchNorm2d(4*hidden_dim_enc),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),
            nn.Conv2d(4*hidden_dim_enc, 8*hidden_dim_enc, kernel_size=3, padding=1, stride=(2,1)), # 32x16 => 16x16
            nn.BatchNorm2d(8*hidden_dim_enc),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),
            nn.Conv2d(8*hidden_dim_enc, 4*hidden_dim_enc, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            nn.BatchNorm2d(4*hidden_dim_enc),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),
            nn.Conv2d(4*hidden_dim_enc, 2*hidden_dim_enc, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            nn.BatchNorm2d(2*hidden_dim_enc),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),
            nn.Flatten(),
            nn.Linear(2*16*hidden_dim_enc, context_dim),
            nn.BatchNorm1d(context_dim),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True)
        )
        
        # decoder z -> X
        self.decoder = nn.Sequential(
            nn.Linear(context_dim, 2*16*hidden_dim_dec),
            nn.BatchNorm1d(2*16*hidden_dim_dec),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),
            View((-1,2*hidden_dim_dec,4,4)),
            nn.ConvTranspose2d(2*hidden_dim_dec, 4*hidden_dim_dec, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            nn.BatchNorm2d(4*hidden_dim_dec),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),
            nn.ConvTranspose2d(4*hidden_dim_dec, 8*hidden_dim_dec, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            nn.BatchNorm2d(8*hidden_dim_dec),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),
            nn.ConvTranspose2d(8*hidden_dim_dec, 4*hidden_dim_dec, kernel_size=3, output_padding=(1,0), padding=1, stride=(2,1)), # 16x16 => 32x16
            nn.BatchNorm2d(4*hidden_dim_dec),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),  
            nn.ConvTranspose2d(4*hidden_dim_dec, 1*hidden_dim_dec, kernel_size=3, output_padding=(1,0), padding=1, stride=(2,1)), # 32x16 => 64x16
            nn.BatchNorm2d(1*hidden_dim_dec),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True),              
            nn.ConvTranspose2d(1*hidden_dim_dec, 1, kernel_size=3, output_padding=(1,0), padding=1, stride=(2,1)), # 64x16 => 128x16
            nn.SiLU(inplace = True),
        )
        
        # linear layer to map the latent space to the context space
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, context_dim),
            nn.BatchNorm1d(context_dim),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True)
        )
        
        # encoder for the context
        self.encoder_c = nn.Sequential(
            nn.Linear(359, 2*context_dim),
            nn.BatchNorm1d(2*context_dim),
            nn.Dropout(drop_prob),
            nn.SiLU(inplace = True), 
            nn.Linear(2*context_dim, context_dim),
        )

        # parameters of the latent distribution
        self.fc_mu = nn.Linear(context_dim, latent_dim)
        self.fc_var = nn.Linear(context_dim, latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
    def configure_optimizers(self):
        """
        Configure the optimizer and the learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_lr)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                }
               }

    def gaussian_likelihood(self, mean, logscale, x) -> torch.Tensor:
        """
        log p(x|z) for gaussian distribution
        Args:
            mean: mean of the gaussian distribution p(z)
            logscale: log of the standard deviation of the gaussian distribution p(z)
            x: input image
        """
        scale = torch.exp(logscale)
        dist = torch.distributions.Laplace(mean, scale)

        # measure prob of seeing image under p(x|z)
        # sum over pixels and channels and average over batch
        return dist.log_prob(x).sum(dim=(-1,-2)).mean(1)

    def kl_divergence(self, z, mu, std) -> torch.Tensor:
        """
        Monte carlo KL divergence
        1. define the first two probabilities (in this case Normal for both)
        2. get the probabilities from the equation

        Args:
            z: latent space
            mu: mean of the gaussian distribution q(z|x)
            std: standard deviation of the gaussian distribution q(z|x)
        """
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        # sum over latent dimension
        kl = (q.log_prob(z) - p.log_prob(z)).sum(-1)
        return kl  
    
    def predict(self, context, n_samples=None, mean=False) -> torch.Tensor:
        """
        Predict the image given the context
        Args:
            context: context vector
            n_samples: number of samples to draw from the latent space
            mean: if True, the mean of the latent space is used, otherwise a sample is drawn
        """
        with torch.no_grad():
            bs = context.shape[0]
            mu = torch.zeros(bs, self.latent_dim).to(self.device)
            std = torch.ones(bs, self.latent_dim).to(self.device)
            pz = torch.distributions.Normal(mu, std) # p(z)            
            # sample from the latent space
            if n_samples != None:
                z = pz.loc.unsqueeze(1).repeat(1, n_samples, 1) if mean else pz.sample([n_samples])
            else:
                z = pz.loc if mean else pz.sample()
            # reshape context according to the number of samples
            if n_samples != None:
                context = context.unsqueeze(1).repeat(1, n_samples, 1)
            else:
                n_samples = 1
            # combine context vector and latent vector
            z = z.reshape(-1, self.latent_dim)
            context = context.reshape(-1, self.context_dim)
            inputs = torch.cat([z, context], 1) #(z,y)
        return self.decoder(inputs).reshape(bs*n_samples, *self.img_dim)
    
    def training_step(self, batch, batch_idx):
        """
        Training step of the model
        """
        elbo, kl, rec_loss = self.step(batch, batch_idx)
        self.log_dict({'kl': kl, 'rec_loss': rec_loss}, prog_bar=True)
        return elbo
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model
        """
        x, context, y = batch
        x_hat = self.predict(context)
        elbo, kl, recon_loss = self.step(batch, batch_idx)
        self.log_dict({
            'val_mae': F.l1_loss(x, x_hat),
            'val_elbo': elbo.item(),
            'val_kl': kl,
            'val_rec_loss': recon_loss
        }, prog_bar=True)
    
    def encode(self, pattern, profile):
        """
        Encode the pattern and the profile to get the latent space
        Args:
            pattern (torch.Tensor): pattern image (bs, 128, 16)
            profile (1d vector): in-plane scattering profile (bs, 359)
        """
        bs = pattern.shape[0]        
        pattern = pattern.unsqueeze(1)
        # encode input data to get p(z|x)
        x_enc = self.encoder(pattern) + self.encoder_c(profile)
        mu, log_var = self.fc_mu(x_enc), self.fc_var(x_enc)
        # sample z from p(z|x)
        std = torch.exp(log_var / 2)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(std).any()
        pz_x = torch.distributions.Normal(mu, std)
        z = pz_x.rsample([self.n_samples]).reshape(-1, self.latent_dim)
        return z, mu, std
    
    def get_context(self, profile, z, n_samples=None):
        """
        get the context vector as a function of the latent space and the in-plane scattering profile
        Args:
            profile (1d vector): in-plane scattering profile
            z (torch.Tensor): latent space
            n_samples (int): number of samples to draw from the latent space
        """
        if n_samples is None:
            n_samples = self.n_samples 
        return self.linear(z) + self.encoder_c(profile).repeat(n_samples, 1)
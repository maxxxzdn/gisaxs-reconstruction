import torch
import torch.nn.functional as F


def minmax(x):
    """
    Normalize a tensor to be in [0,1].
    """
    a = x.min()
    b = x.max()
    if (b-a).item() < 1e-4:
        return torch.zeros_like(x)
    else:
        return (x - a)/(b-a)

def get_lowdim_profile(X):
    """
    Obtain in-plane scattering profile from a compressed 2D GISAXS image.
    """
    assert len(X.shape) == 2
    # interpolate X from 128x16 to 1024x172
    X = F.interpolate(X.unsqueeze(0).unsqueeze(0), (1024,172), mode='bilinear').squeeze()
    # get the region of interest
    x_left = X[341:560, (200-172):(230-172)]
    x_right = X[660:800, (200-172):(230-172)]
    # average over the lateral axis
    x_left = torch.mean(x_left, 1)
    x_right = torch.mean(x_right, 1)
    # normalize
    x_left = minmax(x_left)
    x_right = minmax(x_right)
    return torch.cat([x_left,x_right])

def cvae_log_prob(pipe, profile, params, std=0.1, n_samples=32):
    """
    Computes log p(y|x) for a given in-plane profile x and parameters y.
    Args:
        pipe (Pipeline): joint model including the cVAE and the NFs
        profile (torch.Tensor): in-plane profile
        params (torch.Tensor): parameters of a sample
        std (float): standard deviation of the Gaussian prior (higher covers more of the latent space)
        n_samples (int): number of samples to draw from the prior
    """
    with torch.no_grad():
        # sample from the prior p(z)
        z = std*torch.randn(params.shape[0], n_samples, pipe.cvae.latent_dim)
        z = z.reshape(-1, pipe.cvae.latent_dim).to(profile.device)
        # get the context vector (z||x)
        inputs = pipe.cvae.get_context(profile, z, n_samples)
        params = params.repeat(n_samples, 1)
        return pipe.flow.log_prob(inputs=params, context=inputs)
    
def cvae_log_prob_profile(pipe, pattern, profile, params):
    """
    Computes log p(y|x) for a given GISAXS signal X, in-plane profile x and parameters y.
    Args:
        pipe (Pipeline): pipeline including the cVAE and the NFs
        pattern (torch.Tensor): 2D GISAXS pattern
        profile (torch.Tensor): in-plane profile
        params (torch.Tensor): parameters of a sample
    """
    with torch.no_grad():
        z, _, _ = pipe.cvae.encode(pattern, profile)
        inputs = pipe.cvae.get_context(profile, z, 1)
        return pipe.flow.log_prob(inputs=params, context=inputs)
    
def map_sample(pipe, profile, n_samples, cvae_samples=32):
    """
    Given an in-plane scattering profile, sample from the flow 
    and return parameters of the sample with the highest log probability.
    Args:
        pipe (Pipeline): pipeline
        profile (torch.Tensor): in-plane profile
        n_samples (int): number of samples to draw
        cvae_samples (int): number of samples to draw from the cVAE
    """
    with torch.no_grad():
        # draw z from the latent space of the cVAE (p(z) = N(0, I))
        z = 1*torch.randn(1, cvae_samples, pipe.cvae.latent_dim).to(profile.device).reshape(-1, pipe.cvae.latent_dim)
        # compute the context c = (in-plane profile, z)
        inputs = pipe.cvae.get_context(profile, z, cvae_samples)
        # sample from the flow p(y|c)
        samples, log_probs = pipe.flow.sample_and_log_prob(n_samples, inputs)
        # convert possible nans to negligibly small log-probabilities
        log_probs = torch.nan_to_num(log_probs, -1e5)
        samples = samples.view(-1,72)
        log_probs = log_probs.view(-1)
        # as parameters are normalized to be in [-1,1], we can use this to filter out samples violating the constraints significantly
        log_probs[samples.abs().max(1).values > 1.1] = -1e5
        return samples[log_probs.argmax()], log_probs.max(), inputs.repeat(n_samples,1)[log_probs.argmax()]

import torch
from tqdm import tqdm


class ABC():
    def __init__(self, epsilon, n_params, n_samples= None, mode='2d'):
        """
        Approximate Bayesian Computation (ABC) algorithm to compute p(y|x_ref).
            Data (x,y) is sampled from a dataset (x,y) ~ p(x,y).
            If ||x(y) - x_ref|| < epsilon, then y is accepted, otherwise rejected.
            The accepted y's are used to estimate the posterior p(y|x_ref).
        Args:
            epsilon (float): threshold for distance between patterns
            n_params (int): number of parameters
            n_samples (int): number of samples to be collected
            mode (str): '2d' (2D GISAXS data) or '1d' (in-plane scattering profiles)
        """
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.n_params = n_params
        self.device = 'cpu'
        self.mode = mode
        self.reset()
        
    def reset(self):
        """Reset the ABC algorithm"""
        self.n_searched = 0
        self.out = torch.empty((0, self.n_params)).to(self.device)
        self.best_pattern = None
        self.best_params = None
        self.best_dist = 1e5

    def __len__(self):
        """Returns the number of samples collected"""
        return len(self.out)
    
    def estimate_data(self, data_ref, dataset, verbose=False):
        """
        Estimate the posterior p(y|x_ref) based on collected samples from p(x,y) stored in dataset.
        Args:
            data_ref (torch.Tensor): reference data (x_ref)
            dataset (torch.utils.data.Dataset): dataset of (x,y) samples
            verbose (bool): whether to print progress with tqdm
        """
        assert len(data_ref.shape) == 2
        assert len(dataset[0][0].shape) == 2
        self.reset()
        data_ref = data_ref.unsqueeze(0)
        # assert no samples are collected before the start     
        assert len(self) == 0
        iter_range = tqdm(dataset) if verbose else dataset
        for data in iter_range:
            pattern, profile, params = data
            params = params.unsqueeze(0)

            # preprocess data
            if self.mode == '2d':
                x_ref = data_ref
                x = pattern.unsqueeze(0)
            else:
                x_ref = data_ref.view(-1,1,359)
                x = profile.view(-1,1,359)
            
            batch_dist = self.l2_distance(x, x_ref)
            mask = batch_dist < self.epsilon
            params_ = params[mask] # x that are sufficiently close to ref
            self.out = torch.cat([self.out, params_])
            self.n_searched += 1
            
            best_batch_idx = batch_dist.argmax()
            if self.best_dist > batch_dist[best_batch_idx]:
                self.best_x = x[best_batch_idx]
                self.best_params = params[best_batch_idx]
                self.best_dist = batch_dist[best_batch_idx]
                         
    @staticmethod
    def l2_distance(x, y):
        """Returns L2 distance between N images"""
        assert len(x.shape) == len(y.shape), print(x.shape, y.shape)
        return (x - y).pow(2).sum((-1,-2)).sqrt()
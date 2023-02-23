import torchmetrics
import numpy as np
from numpy import prod
import pytorch_lightning as pl
from torch import optim, nn, Tensor
from typing import Tuple

from .utils import View


class Surrogate(pl.LightningModule):
    def __init__(self, model, loss_name, lr):
        """
        Defines the surrogate model.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = self._loss(loss_name)
        self.train_mse, self.val_mse = torchmetrics.MeanSquaredError(), torchmetrics.MeanSquaredError()
        self.train_mae, self.val_mae = torchmetrics.MeanAbsoluteError(), torchmetrics.MeanAbsoluteError()
        
    def forward(self, x):
        """
        Forward pass of the surrogate model.
        """
        return self.model(x)

    def training_step(self, batch:list, batch_idx: int):
        """
        Training step of the surrogate model.
        """
        pattern, params = batch
        pattern_hat = self(params)
        loss = self.loss(pattern_hat, pattern)
        self.train_mse(pattern_hat, pattern)
        self.train_mae(pattern_hat, pattern)
        return loss
    
    def training_epoch_end(self, training_step_outputs: list):
        """
        Aggregate metrics output over epoch and log for training data.
        """
        self.log_dict({'train_mse_epoch': self.train_mse, 'train_mae_epoch': self.train_mae})
    
    def validation_step(self, batch:list, batch_idx: int):
        """
        Validation step of the surrogate model.
        """
        pattern, params = batch
        pattern_hat = self.model(params)
        loss = self.loss(pattern_hat, pattern)
        self.val_mse(pattern_hat, pattern)
        self.val_mae(pattern_hat, pattern)
        return loss
    
    def validation_epoch_end(self, validation_step_outputs: list):
        """
        Aggregate metrics output over epoch and log for validation data.
        """
        self.log_dict({'val_mse_epoch': self.val_mse, 'val_mae_epoch': self.val_mae})

    def configure_optimizers(self):
        """
        Defines model optimizer.
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler, 
                                 "monitor": "train_mse_epoch"}}
    
    @staticmethod
    def _loss(loss_name):
        """
        Defines loss function.
        """
        if loss_name == 'l1':
            return nn.L1Loss()
        elif loss_name == 'l2':
            return nn.MSELoss()
        else:
            raise NotImplementedError("Only 'l1' and 'l2' options are supported.")
            
            
class Surrogate1D(Surrogate):
    """
    Defines a 1D surrogate model which works with in-plane signal.
    Input:
        n_params (int): number of parameters in a sample
        out_dim (int): number of output channels if a signal
        loss_name (str): loss function name
        lr (float): learning rate

    """
    def __init__(self, n_params: int, out_dim: int, 
                 loss_name: str, lr: float):
        model = nn.Sequential(
            nn.Linear(n_params,64), nn.ReLU(), 
            nn.Linear(64,128), nn.ReLU(), 
            nn.Linear(128,512), nn.ReLU(), 
            nn.Linear(512, out_dim))        
        super().__init__(model, loss_name, lr)
        self.save_hyperparameters()
        
class Surrogate2D(Surrogate):
    """
    Defines a 2D surrogate model which works with the image signal.
    Input:
        n_params (int): number of parameters in a sample
        out_dim (Tuple[int, int]): output dimensions of a signal
        loss_name (str): loss function name
        lr (float): learning rate
        drop_prob (float): dropout probability
    """
    def __init__(self, n_params: int, hidden_dim: int, out_dim: Tuple[int, int], 
                 loss_name: str, lr: float, drop_prob: float):
        model = nn.Sequential(
            nn.Linear(n_params, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace = True),
            nn.Dropout(drop_prob),
            
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.SiLU(inplace = True),
            nn.Dropout(drop_prob),
            
            nn.Linear(2*hidden_dim, 4*hidden_dim), 
            nn.BatchNorm1d(4*hidden_dim),
            nn.SiLU(inplace = True),
            nn.Dropout(drop_prob),
            
            nn.Linear(4*hidden_dim, prod(out_dim)),
            View((-1, out_dim[0], out_dim[1])),
        )
       
        super().__init__(model, loss_name, lr) 
        self.save_hyperparameters()
        
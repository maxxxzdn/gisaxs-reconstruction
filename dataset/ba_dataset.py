import os
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from glob import glob
import torch.nn.functional as F
from torch.nn.functional import dropout
from torch.utils.data import Dataset, DataLoader, random_split

from .utils import read_single_hdf5
from .preprocessing import salt_and_pepper, Transform

    
class BaDataset(Dataset):
    def __init__(self, path, indices, to_preload, to_augment, in_shape, out_shape, verbose=True, **kwargs):
        super().__init__()
        kwargs.setdefault("sigma", 0.)
        kwargs.setdefault("drop_y", 0.)
        kwargs.setdefault("sp_prob", 0.)
        kwargs.setdefault("mask", False)
        kwargs.setdefault("log", True)
        kwargs.setdefault("minmax", True)
        kwargs.setdefault("equalize", False)
        
        self.path = path
        self.indices = indices
        self.to_augment = to_augment
        self.to_preload = to_preload
        self.verbose = verbose
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.transform = Transform(kwargs['log'], kwargs['minmax'], kwargs['equalize'], 
                                   in_shape, out_shape)
        
        self.sigma = kwargs['sigma']
        self.drop_y = kwargs['drop_y']
        self.sp_prob = kwargs['sp_prob']
        
        self.X = []
        self.y = []
       
        self.preloaded = False
        if to_preload:
            self._preload()
            self.preloaded = True
            self.indices = range(len(self))
            
    def get_mask(self):
        mask = torch.ones(128)
        mask[0:10] = 0
        mask[60:80] = 0
        mask[-20:] = 0
        return mask.unsqueeze(0) 
    
    def perturb_proj(self, x):
        return x*self.mask
            
    def _preload(self):
        seq = range(len(self))
        if self.verbose:
            seq = tqdm(seq)
        for index in seq:
            X,y = self[index]
            self.X.append(X)
            self.y.append(y)
              
    def __getitem__(self, index):
        if self.preloaded:
            X, y = self.X[index], self.y[index]
        else:
            X, y = self._read(index)
            X = self.transform(X)
        if self.to_augment: 
            # do not augment when preloading
            if self.to_preload and not self.preloaded:
                pass 
            else:
                X, y = self._augment(X,y)
        return X.float(), y.float()

    def _read(self, index):
        index_ = self.indices[index]
        try:
            X, y = read_single_hdf5(index_, self.path)
        except:
            print(f'File {index_}.h5 is not found')
        X = torch.Tensor(X.reshape(*self.in_shape))
        y = torch.Tensor(y.reshape(-1))
        return X, y
    
    def _augment(self, X, y,):
        X = self._augment_X(X)
        y = self._augment_y(y)  
        return X, y
            
    def _augment_X(self, X):
        if self.sigma > 0:
            X += self.sigma*torch.randn_like(X)
        if self.sp_prob > 0:
            X = salt_and_pepper(X, self.sp_prob)
        return X
            
    def _augment_y(self, y):
        return dropout(y, self.drop_y)

    def __len__(self):
        return len(self.indices)
    
    @staticmethod
    def _project(X):
        assert len(X.shape) == 2
        return X.mean(1)
    
class BaDataset1D(BaDataset):
    def __init__(self, path, indices, to_preload, to_augment, in_shape, out_shape, verbose=True, **kwargs):
        super().__init__(path, indices, False, to_augment, in_shape, out_shape, verbose, **kwargs)
        if to_preload:
            self._preload()
            self.preloaded = True
    
    def __getitem__(self, index):
        if self.preloaded:
            X, y = self.X[index], self.y[index]
            if self.to_augment:
                X, y = self._augment(X,y)
            X = self._project(X)
        else:
            X, y = self._read(index)
            X = self.transform(X)
        return X, y
    

class BaDataset1D2D(BaDataset):
    def __init__(self, path, indices, to_preload, to_augment, in_shape, out_shape, verbose=True, **kwargs):
        super().__init__(path, indices, False, to_augment, in_shape, out_shape, verbose, **kwargs)
        self.mask = 1. if kwargs['mask'] is False else self.get_mask()
        length = len(indices)
        if kwargs['order']:
            self.indices = indices
        else:
            self.indices = [int(x.split('/')[-1].split('.')[0]) for x in glob(path + '*')][:length]
        self.x = []
        if to_preload:
            self._preload()
            self.preloaded = True
        self.indices = range(len(self))
            
    def get_mask(self):
        mask = torch.zeros(self.in_shape[0])
        mask[341:560] = 1.
        mask[660:800] = 1.
        return mask.unsqueeze(0)
            
    def _augment(self, X, x, y):
        X = self._augment_X(X)
        x = self._augment_x(x)
        y = self._augment_y(y) 
        return X, x, y
    
    def _augment_x(self, x):
        x = self.mask*x
        return X
    
    def _project(self, X):
        assert len(X.shape) == 2
        x_left = X[341:560, 200:230]
        x_right = X[660:800, 200:230]
        x_left = torch.mean(x_left, 1)
        x_right = torch.mean(x_right, 1)
        x_left = self.minmax(x_left)
        x_right = self.minmax(x_right)
        return torch.cat([x_left,x_right])
        
    @staticmethod
    def minmax(x):
        a = x.min()
        b = x.max()
        if (b-a).item() < 1e-4:
            return torch.zeros_like(x)
        else:
            return (x - a)/(b-a)
              
    def __getitem__(self, index):
        if self.preloaded:
            X, x, y = self.X[index], self.x[index], self.y[index]
            if self.to_augment:
                X, x, y = self._augment(X, x, y)
        else:
            X, y = self._read(index)
            x = self._project(X.reshape(1024,512))
            X = self.transform(X)
            X = F.interpolate(X.unsqueeze(1), self.out_shape).squeeze()
        assert torch.allclose(X, X)
        assert torch.allclose(x, x)
        return X.float(), x.float(), y.float()            

    def _preload(self):
        seq = range(len(self))
        if self.verbose:
            seq = tqdm(seq)
        for index in seq:
            X, x, y = self[index]
            self.X.append(X)
            self.x.append(x)
            self.y.append(y)    

class GISAXSDataModule(pl.LightningDataModule):
    def __init__(self, mode: str, batch_size: int, **kwargs):
        super().__init__()
        assert mode in ['1d', '2d', '1d2d']
        if mode == '1d':
            self._dataset = BaDataset1D 
        elif mode == '2d':
            self._dataset = BaDataset
        else:
            self._dataset = BaDataset1D2D
        self.batch_size = batch_size
        self.kwargs = kwargs

    def setup(self, stage=None):
        gisaxs_full = torch.load('/home/zhdano82/aiGISAXS/data_100.pt') #self._dataset(**self.kwargs)
        #self.gisaxs_full = gisaxs_full = self._dataset(**self.kwargs)
        train_size = int(0.8 * len(gisaxs_full))
        test_size = len(gisaxs_full) - train_size
        self.gisaxs_train, self.gisaxs_val = random_split(gisaxs_full, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.gisaxs_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.gisaxs_val, batch_size=self.batch_size)

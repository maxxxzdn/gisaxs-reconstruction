import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from glob import glob
import torch.nn.functional as F
from torch.nn.functional import dropout
from torch.utils.data import Dataset, DataLoader, random_split

from .utils import read_single_hdf5
from .preprocessing import salt_and_pepper, Transform

    
class BaDataset(Dataset):
    """
    Custom dataset for the preprocessed BornAgain data
    Inputs:
        path (str): path to the data
        indices (list): list of indices to use
        to_preload (bool): whether to preload the data
        to_augment (bool): whether to augment the data
        in_shape (tuple): shape of the input
        out_shape (tuple): shape of the output
        verbose (bool): whether to show progress bar
        **kwargs: additional arguments related to preprocessing and augmentation
    """
    def __init__(self, path, indices, to_preload, to_augment, in_shape, out_shape, verbose=True, **kwargs):
        super().__init__()
        # set default values for preprocessing and augmentation
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
            
    def calibrate(self):
        """
        Calibrate the dataset
        """
        # indices of non-empty data
        nons = [x != None for x in self.X]
        # remove empty data
        self.indices = np.array(self.indices)[np.where(nons)[0]]
        self.X = np.array(self.X)[np.where(nons)[0]]
        self.y = np.array(self.y)[np.where(nons)[0]]
        self.x = np.array(self.x)[np.where(nons)[0]]
            
    def get_mask(self):
        pass
    
    def perturb_proj(self, x):
        """
        Perturb the projection
        Inputs:
            x (torch.Tensor): input projection
        """
        return x*self.mask
            
    def _preload(self):
        """
        Preload the data
        """
        seq = range(len(self))
        if self.verbose:
            seq = tqdm(seq)
        for index in seq:
            X,y = self[index]
            self.X.append(X)
            self.y.append(y)
              
    def __getitem__(self, index):
        """
        If preloading was done, return the preloaded data
        Otherwise, read the data from the disk
        Augmentation is disabled when reading the data from the disk (i.e. when preloading)
        """
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
        """
        Read the data from the disk
        """
        index_ = self.indices[index]
        try:
            X, y = read_single_hdf5(index_, self.path)
        except:
            raise FileNotFoundError(f'File {index_}.h5 is not found')
        X = torch.Tensor(X.reshape(*self.in_shape))
        y = torch.Tensor(y.reshape(-1))
        return X, y
    
    def _augment(self, X, y,):
        """
        Augment the data
        Inputs:
            X (torch.Tensor): image data
            y (torch.Tensor): parameter data
        """
        X = self._augment_X(X)
        y = self._augment_y(y)  
        return X, y
            
    def _augment_X(self, X):
        """
        Augment the image data
        Applies Gaussian noise and salt-and-pepper noise
        """
        if self.sigma > 0:
            X += self.sigma*torch.randn_like(X)
        if self.sp_prob > 0:
            X = salt_and_pepper(X, self.sp_prob)
        return X
            
    def _augment_y(self, y):
        """
        Augment the parameter data (dropout)
        """
        return dropout(y, self.drop_y)

    def __len__(self):
        return len(self.indices)
    
    @staticmethod
    def _project(X):
        """
        Compute in-plane projection by averaging over the out-of-plane axis
        """
        assert len(X.shape) == 2
        return X.mean(1)
    
class BaDataset1D(BaDataset):
    """
    Custom dataset for the preprocessed BornAgain data that has ONLY in-plane projections
    See the documentation of BaDataset for the details
    """
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
    """
    Custom dataset for the preprocessed BornAgain data that has both in-plane projections and 2D images
    See the documentation of BaDataset for the details
    One can pass the 'order' argument to the constructor to specify the order of the data (random or sequential)
    """
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
        """
        Mask for the in-plane signal to make it consistent with Lisa's analysis
        Source: Lisa Randolph (lisa.randolph@xfel.eu)
        """
        mask = torch.zeros(self.in_shape[0])
        mask[341:560] = 1.
        mask[660:800] = 1.
        return mask.unsqueeze(0)
            
    def _augment(self, X, x, y):
        """
        Augment the image and its parameters, mask its projection
        """
        X = self._augment_X(X)
        y = self._augment_y(y) 
        return X, self.mask*x, y
        
    def _project(self, X):
        """
        Extract the in-plane signal from the image
        Source: Lisa Randolph (lisa.randolph@xfel.eu)
        """
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
        """
        Min-max normalization
        """
        a = x.min()
        b = x.max()
        if (b-a).item() < 1e-4:
            return torch.zeros_like(x)
        else:
            return (x - a)/(b-a)
              
    def __getitem__(self, index):
        """
        See the documentation of BaDataset for the details
        Asserts that neither 2D not 1D data is corrupted
        """
        if self.preloaded:
            X, x, y = self.X[index], self.x[index], self.y[index]
            assert X is not None, f'Corrupted 2D data at index {index}'
            if self.to_augment:
                X, x, y = self._augment(X, x, y)
            X, x, y = X.float(), x.float(), y.float() 
        else:
            X, y = self._read(index)
            x = self._project(X.reshape(1024,512))
            X = self.transform(X)
            X = F.interpolate(X.unsqueeze(1), self.out_shape).squeeze() #, mode='bilinear').squeeze()
        assert torch.allclose(X, X), f'Corrupted 2D data at index {index}'
        assert torch.allclose(x, x), f'Corrupted 1D data at index {index}'
        return X.float(), x.float(), y.float()            

    def _preload(self):
        """
        Preload the data to RAM
        """
        seq = range(len(self))
        if self.verbose:
            seq = tqdm(seq)
        for index in seq:
            X, x, y = self[index]
            self.X.append(X)
            self.x.append(x)
            self.y.append(y)    

class GISAXSDataModule(pl.LightningDataModule):
    """
    Pytorch lightning DataModule for the BornAgain data
    It is used to load the data and split it into train and test sets
    Input:
        mode: '1d', '2d' or '1d2d' - the type of the data to load
        batch_size: the size of the batch
        kwargs: the arguments to pass to the dataset constructor
    """
    def __init__(self, mode: str, batch_size: int, preloaded_files: dict=None, **kwargs):
        super().__init__()
        assert mode in ['1d', '2d', '1d2d']
        if mode == '1d':
            self._dataset = BaDataset1D 
        elif mode == '2d':
            self._dataset = BaDataset
        else:
            self._dataset = BaDataset1D2D
        self.batch_size = batch_size
        self.preloaded_files = preloaded_files
        self.kwargs = kwargs

    def setup(self, stage=None):
        """
        Split the data into train and test sets
        Input:
            stage: the stage of the training (not used)
        """
        if self.preloaded_files['train'] is None:
            gisaxs_full = self._dataset(**self.kwargs)
        else:
            gisaxs_full = torch.load(self.preloaded_files['train'])
            gisaxs_full.calibrate()
        train_size = int(0.8 * len(gisaxs_full))
        test_size = len(gisaxs_full) - train_size
        self.gisaxs_train, self.gisaxs_val = random_split(gisaxs_full, [train_size, test_size])

    def train_dataloader(self):
        """
        Return the train dataloader
        """
        return DataLoader(self.gisaxs_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """
        Return the validation dataloader
        """
        return DataLoader(self.gisaxs_val, batch_size=self.batch_size)

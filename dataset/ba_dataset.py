from tqdm import tqdm
from glob import glob
from torch import tensor
from torch.nn.functional import dropout
from .utils import read_single_hdf5
from torch.utils.data import Dataset, DataLoader

# reproducibility
import torch
#torch.manual_seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    
def salt_and_pepper(x_in, prob):
    x_out = x_in.clone()
    noise_tensor=torch.rand_like(x_out)
    salt=torch.max(x_out)
    pepper=torch.min(x_out)
    x_out[noise_tensor < prob/2]=salt
    x_out[noise_tensor > 1-prob/2]=pepper
    return x_out

class BaDataset(Dataset):
    def __init__(self, path, indices, preload, transform, augmentation, sigma, drop_y, sp_prob, verbose=True):
        super().__init__()
        self.path = path
        self.indices = indices
        self.to_preload = preload
        self.preloaded = False
        self.transform = transform
        self.augmentation = augmentation
        self.sigma = sigma
        self.drop_y = drop_y
        self.sp_prob = sp_prob
        self.verbose = verbose
        
        if self.to_preload:
            self.X = []
            self.y = []
            self.preload()
        
    def __getitem__(self, index):
        if self.preloaded:
            X, y = self.X[index], self.y[index]
        else:
            index_ = self.indices[index]
            try:
                X, y = read_single_hdf5(index_, self.path)
                X = tensor(X.reshape(1200, 120)).float()
                y = tensor(y.reshape(-1)).float()
                X = self.transform(X)
            except:
                print(f'File {index_}.h5 is not found')
        if self.augmentation:
            X = self.augment_X(X)
            y = self.augment_y(y)                    
        return X, y
    
    def preload(self):
        seq = range(len(self))
        if self.verbose:
            seq = tqdm(seq)
        for index in seq:
            X,y = self[index]
            self.X.append(X)
            self.y.append(y)
        self.preloaded = True
        
    #def augment_X(self, X):
    #    if self.sigma > 0:
    #        X += self.sigma*torch.randn_like(X)
    #    if self.sp_prob > 0:
    #        X = salt_and_pepper(X, self.sp_prob)
    #    return X
            
    #def augment_y(self, y):
    #    return dropout(y, self.drop_y)

    def __len__(self):
        return len(self.indices)

    
def setup_data_loaders(path, n_samples, batch_size, transform, 
                       augmentation, sigma, drop_y, sp_prob,
                       val_frac=0.2, preload=False, start_id=0, 
                       verbose=True, **kwargs):
    """
     helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param val_frac: fraction of validation data examples
    :param kwargs: other params for the pytorch data loader
    :return: two data loaders: (data for training, data for validation)
    """

    cached_data = {}
    loaders = {}
    
    all_indices = [int(file.split('/')[-1].split('.')[0]) for file in glob(path + '*')]
    all_indices = all_indices[start_id:n_samples]
    
    split = int((1 - val_frac) * len(all_indices))
    indices = {"train": all_indices[:split] , "val": all_indices[split:]}
    
    #g = torch.Generator()
    #g.manual_seed(0)

    for mode in ["train", "val"]:
        cached_data[mode] = BaDataset(path, indices[mode], preload, transform, 
                                      augmentation, sigma, drop_y, sp_prob, verbose)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, 
                                   shuffle=True) #, worker_init_fn=seed_worker, 
                                   #generator=g, **kwargs)
        
    print("\n Data loaders created")
    return loaders
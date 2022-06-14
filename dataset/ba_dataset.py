from tqdm import tqdm
from torch import tensor
from .utils import read_single_hdf5
from torch.utils.data import Dataset, DataLoader

class BaDataset(Dataset):
    def __init__(self, index_range, path, preload, transform, *args, **kwargs):
        super().__init__()
        self.index_range = index_range
        self.path = path
        self.preloaded = False
        self.transform = transform
        if preload:
            self.X = []
            self.y = []
            self.preload()
        
    def __getitem__(self, index):
        if self.preloaded:
            return self.X[index], self.y[index]
        else:
            index_ = self.index_range[index]
            try:
                X, y = read_single_hdf5(index_, self.path)
                X = tensor(X.reshape(1200, 120)).float()
                y = tensor(y.reshape(-1)).float()
                return self.transform(X), y
            except:
                print(f'File {index_}.h5 is not found')
    
    def preload(self):
        for index in tqdm(range(len(self.index_range))):
            X,y = self[index]
            self.X.append(X)
            self.y.append(y)
        self.preloaded = True

    def __len__(self):
        return len(self.index_range)

    
def setup_data_loaders(path, n_samples, batch_size, transform, val_frac=0.2, preload=False, **kwargs):
    """
     helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param val_frac: fraction of validation data examples
    :param kwargs: other params for the pytorch data loader
    :return: two data loaders: (data for training, data for validation)
    """

    cached_data = {}
    loaders = {}
    
    modes = ["train", "val"]
    split = int((1 - val_frac) * n_samples)
    ranges = {"train": range(n_samples)[:split] , "val": range(n_samples)[split:]}

    for mode in modes:
        cached_data[mode] = BaDataset(ranges[mode], path, preload, transform)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs)
        
    print("Data loaders created")
    return loaders
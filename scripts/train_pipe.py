import sys
sys.path.append('..')
import torch
import pytorch_lightning as pl
from datetime import datetime

from dataset.ba_dataset import GISAXSDataModule
from models.pipeline import Pipeline


mode = '1d2d' # both in-plane projections and 2D images
path = '/bigdata/hplsim/aipp/Maksim/BA_simulation/complete/'
batch_size = 32
train_frac = 0.8 # fraction of points to train on, remaining - to validate on
n_layers = 12 # number of layers in material

#! change n_dp tp 650000 when training the final model
n_dp = 10000 # number of datapoints in the training dataset

in_shape = (1024,512) # shape of images in raw data
out_shape = (128,16) # shape of preprocessed images
to_preload = False # preload data in RAM before training

# augmentation variables
to_augment = 0
sigma = None
drop_y = None
sp_prob = None
drop_prob = None

train_file = f'/bigdata/hplsim/aipp/Maksim/BA_simulation/exp_data/data_{n_layers}_{n_dp}.pt'
test_file = f'/bigdata/hplsim/aipp/Maksim/BA_simulation/exp_data/data_{n_layers}_test.pt'
preloaded_files = {'train': train_file, 'test_file': test_file}

indices = range(0, n_dp)
data_module = GISAXSDataModule(mode, batch_size, preloaded_files=preloaded_files, path=path, 
                               indices=indices, to_preload=to_preload, to_augment=to_augment,
                               in_shape=in_shape, out_shape=out_shape,
                               sigma=sigma, drop_y=drop_y, sp_prob=sp_prob, 
                               mask=True, verbose=True, order=False)

config = dict(
    context_dim=96,
    flow_hidden=64,
    hidden_dim_dec=4,
    hidden_dim_enc=16,
    latent_dim=4)

cvae_params = dict(
    latent_dim=config['latent_dim'],
    context_dim=config['context_dim'], 
    hidden_dim_enc=config['hidden_dim_enc'],
    hidden_dim_dec=config['hidden_dim_dec'],
    n_samples=1,
    drop_prob=0.1,
)

pipe = Pipeline(n_layers=12, n_transforms=8, hidden_dim=config['flow_hidden'], 
                cvae_params=cvae_params, lr=1e-3, step_lr=10)

progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=1000)

trainer = pl.Trainer(logger=None, max_epochs=30, 
                     callbacks=progress_bar, devices="auto", accelerator="auto", 
                     enable_progress_bar=True, enable_checkpointing=False, 
                     gradient_clip_val=1.5, accumulate_grad_batches=1)

trainer.fit(model=pipe, datamodule=data_module)

dateTimeObj = datetime.now()
timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
torch.save(pipe.state_dict(), f'../saved_models/pipe_{n_dp}_{timestring}.pt')
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser

from models import Surrogate1D, Surrogate2D
from dataset.ba_dataset import GISAXSDataModule

parser = ArgumentParser()
# dataset
parser.add_argument("--n_layers", type = int)
parser.add_argument("--start_id", type = int)
parser.add_argument("--end_id", type = int)
parser.add_argument("--to_augment", type = int)
parser.add_argument("--sigma", type = float, default = 0.01)
parser.add_argument("--drop_y", type = str, default = 0.05)
parser.add_argument("--sp_prob", type = str, default = 0.01)
# model
parser.add_argument("--hidden_dim", type = int)
parser.add_argument("--drop_prob", type = float, default = 0.0)
# optimization
parser.add_argument("--n_epochs", type = int)
parser.add_argument("--lr", type = float, default = 5e-4)
parser.add_argument("--batch_size", type = int, default = 32)
parser.add_argument("--loss", type = str, default = 'l2')
# other
parser.add_argument("--verbose", type = int, default=0)
args = parser.parse_args()


indices = range(args.start_id, args.end_id)
path = '/bigdata/hplsim/aipp/Maksim/BA_simulation/layer_{}/'.format(args.n_layers)
if args.n_layers < 12:
    in_shape = (1200,120)
    out_dim = (128,16)
else:
    in_shape = (1024,512)
    out_dim = (64,32)

model = Surrogate2D(n_params=args.n_layers*6, hidden_dim = args.hidden_dim, 
                    out_dim=out_dim, loss_name=args.loss, lr=args.lr, drop_prob=args.drop_prob)
data_module = GISAXSDataModule(mode='2d', batch_size=args.batch_size,
                               path=path, indices=range(args.start_id, args.end_id), 
                               to_preload=True, to_augment=args.to_augment,
                               in_shape=in_shape, out_shape=out_dim, verbose=args.verbose,
                               sigma=args.sigma, drop_y=args.drop_y, sp_prob=args.sp_prob)

#logger = CSVLogger("../csv_logs", name="{}_surrogate".format(args.mode))
wandb_logger = WandbLogger(project="GISAXS", log_model="all")
wandb_logger.experiment.config.update(args)
wandb_logger.watch(model)
checkpoint_callback = ModelCheckpoint(monitor="val_mse_epoch", mode="max")
    
trainer = pl.Trainer(logger=wandb_logger, max_epochs=args.n_epochs, 
                     callbacks=[checkpoint_callback],
                     devices="auto", accelerator="auto", 
                     enable_progress_bar=False, enable_checkpointing=True, 
                     default_root_dir="../saved_models/", gradient_clip_val=1.5, 
                     log_every_n_steps=1000)

trainer.fit(model=model, datamodule=data_module)

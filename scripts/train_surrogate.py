import sys
sys.path.append('..')

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
from argparse import ArgumentParser

from models import Surrogate1D, Surrogate2D
from dataset.ba_dataset import GISAXSDataModule

parser = ArgumentParser()
parser.add_argument("--mode", type = str)
parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--lr", type = float, default = 5e-4)
parser.add_argument("--start_id", type = int)
parser.add_argument("--end_id", type = int)
parser.add_argument("--n_layers", type = int)
parser.add_argument("--to_augment", type = int)
parser.add_argument("--sigma", type = float, default = 0.01)
parser.add_argument("--drop_y", type = str, default = 0.05)
parser.add_argument("--sp_prob", type = str, default = 0.01)
parser.add_argument("--drop_prob", type = float, default = 0.0)
parser.add_argument("--n_epochs", type = int)
parser.add_argument("--loss", type = str, default = 'l2')
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
    
if args.mode == '1d':
    surrogate = Surrogate1D
    out_dim = out_dim[0]
elif args.mode == '2d' or args.mode == '1d2d':
    surrogate = Surrogate2D
else:
    raise NotImplementedError

model = surrogate(n_params=args.n_layers*6, out_dim=out_dim, loss_name=args.loss, lr=args.lr, drop_prob=args.drop_prob)
data_module = GISAXSDataModule(mode=args.mode, batch_size=args.batch_size,
                               path=path, indices=range(args.start_id, args.end_id), 
                               to_preload=True, to_augment=args.to_augment,
                               in_shape=in_shape, out_shape=out_dim, verbose=args.verbose,
                               sigma=args.sigma, drop_y=args.drop_y, sp_prob=args.sp_prob)

logger = CSVLogger("../csv_logs", name="{}_surrogate".format(args.mode))
trainer = pl.Trainer(logger=logger, max_epochs=args.n_epochs, 
                     devices="auto", accelerator="auto", 
                     enable_progress_bar=False, enable_checkpointing=True, 
                     default_root_dir="../saved_models/", gradient_clip_val=1.5, 
                     log_every_n_steps=1000)

trainer.fit(model=model, datamodule=data_module)

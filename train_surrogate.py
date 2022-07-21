# reproducibility
import torch
torch.manual_seed(0)

from torch.optim import AdamW
from torch.nn import L1Loss, MSELoss
from argparse import ArgumentParser

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

from models.convnet import ConvNet
from models.fcnet import FCNet
from models.resnet import ResNet
from models.training import train, L1SSIM

from dataset.ba_dataset import setup_data_loaders
from dataset.preprocessing import Transform

parser = ArgumentParser()
#dataset
parser.add_argument("--n_samples", type = int, required=True)
parser.add_argument("--batch_size", type = int, default = 100)
parser.add_argument("--n_layers", type = int, default = 3)
parser.add_argument("--data_path", type = str, required=True)
#augmentation
parser.add_argument("--augmentation", type = int, required=True)
parser.add_argument("--sigma", type = float, default = 0.01)
parser.add_argument("--drop_y", type = float, default = 0.05)
parser.add_argument("--sp_prob", type = float, default = 0.01)
#preprocessing
parser.add_argument("--in_shape", type = int, nargs='+', required=True)
parser.add_argument("--log", type = int, default = 1)
parser.add_argument("--minmax", type = int, default = 1)
parser.add_argument("--equalize", type = int, default = 0)
#model
parser.add_argument("--model", type = str, default = 'fcnet')
parser.add_argument("--n_channels", type = int, default = 32)
parser.add_argument("--kernel_size", type = int, default = 3)
parser.add_argument("--mode", type = str, default = 'transpose', choices = ['upsample', 'transpose'])
parser.add_argument("--drop_model", type = float, default = 0.1)
#optimization
parser.add_argument("--learning_rate", type = float, default = 1e-4)
parser.add_argument("--distance", type = str, default = 'l1', choices = ['l1', 'l2', 'l1_ssim'])
parser.add_argument("--l1_weight", type = float, default = 0.5)
parser.add_argument("--window_size", type = int, nargs='+', required=True)
parser.add_argument("--n_epochs", type = int, default = 1000)
parser.add_argument("--train", type = int, default = 1)
args = parser.parse_args()

#log, minmax, equalize
transformation = Transform(args.in_shape, args.log, args.minmax, args.equalize)
loaders = setup_data_loaders(
    args.data_path , args.n_samples, args.batch_size, transform=transformation, 
    augmentation=args.augmentation, sigma=args.sigma, drop_y=args.drop_y, sp_prob=args.sp_prob, 
    val_frac = 0.05, preload=True)

name = '_'.join([args.model, 
                 str(args.n_channels) if args.model != 'fcnet' else '', 
                 args.mode if args.model != 'fcnet' else '', 
                 str(args.log), str(args.minmax), str(args.equalize), 
                 args.distance, str(args.n_layers), '_'.join(str(x) for x in args.in_shape)])
if args.model == 'convnet':
    model = ConvNet(args.in_shape, 6*args.n_layers, args.n_channels, args.kernel_size, args.mode, name).cuda()
elif args.model == 'resnet':
    model = ResNet(6*args.n_layers, args.n_channels, name).cuda()
elif args.model == 'fcnet':
    model = FCNet(6*args.n_layers, name, args.drop_model).cuda()
else:
    raise NotImplementedError

optimizer = AdamW(model.parameters(), lr = args.learning_rate)  
if args.distance == 'l1':
    loss_func = L1Loss()
elif args.distance == 'l2':
    loss_func = MSELoss()
elif args.distance == 'l1_ssim':
    loss_func = L1SSIM(kernel_size=args.window_size, weight = args.l1_weight)
else:
    raise NotImplementedError
    
if args.train:
    train(args.n_epochs, model, loaders, loss_func, optimizer, './saved_models/')

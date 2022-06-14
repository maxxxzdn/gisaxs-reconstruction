from torch.optim import AdamW
from torch.nn import L1Loss, MSELoss
from argparse import ArgumentParser

from models.convnet import ConvNet
from models.resnet import ResNet
from models.training import train

from dataset.ba_dataset import setup_data_loaders
from dataset.preprocessing import Transform

parser = ArgumentParser()
#dataset
parser.add_argument("--n_samples", type = int)
parser.add_argument("--batch_size", type = int, default = 100)
parser.add_argument("--n_layers", type = int, default = 3)
parser.add_argument("--data_path", type = str)
parser.add_argument("--data_min", type = float)
parser.add_argument("--data_max", type = float)
#preprocessing
parser.add_argument("--log", type = int, default = 0)
parser.add_argument("--minmax", type = int, default = 1)
parser.add_argument("--equalize", type = int, default = 0)
#model
parser.add_argument("--model", type = str, default = 'convnet')
parser.add_argument("--n_channels", type = int, default = 32)
parser.add_argument("--kernel_size", type = int, default = 3)
parser.add_argument("--mode", type = str, default = 'transpose', choices = ['upsample', 'transpose'])
#optimization
parser.add_argument("--learning_rate", type = float, default = 1e-4)
parser.add_argument("--distance", type = str, default = 'l2', choices = ['l1', 'l2'])
parser.add_argument("--n_epochs", type = int, default = 100)
args = parser.parse_args()

#log, minmax, equalize
transformation = Transform(args.log, args.minmax, args.equalize, args.data_min, args.data_max)
loaders = setup_data_loaders(
    args.data_path , args.n_samples, args.batch_size, 
    transform=transformation, val_frac = 0.05, preload=True)

if args.model == 'convnet':
    name = '_'.join([args.model, str(args.n_channels), args.mode, 
                     str(args.log), str(args.minmax), str(args.equalize), 
                     args.distance])
    model = ConvNet(6*args.n_layers, args.n_channels, args.kernel_size, args.mode, name).cuda()
elif args.model == 'resnet':
    model = ResNet(6*args.n_layers, args.n_channels).cuda()
else:
    raise NotImplementedError
    
optimizer = AdamW(model.parameters(), lr = args.learning_rate)  
if args.distance == 'l1':
    loss_func = L1Loss()
elif args.distance == 'l2':
    loss_func = MSELoss()
else:
    raise NotImplementedError

train(args.n_epochs, model, loaders, loss_func, optimizer, './saved_models/')

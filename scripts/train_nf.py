from torch import load
from torch.nn import Linear
from torch.optim import AdamW
from argparse import ArgumentParser

from models.convnet import ConvNet
from models.fcnet import FCNet
from models.resnet import ResNet
from models.training import train, train_nf
from models.nf_utils import Logit
from models.encoder import Encoder

from dataset.ba_dataset import setup_data_loaders
from dataset.preprocessing import Transform

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


parser = ArgumentParser()
parser.add_argument("--savedir", type = str, default = None)
#dataset
parser.add_argument("--n_samples", type = int)
parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--n_layers", type = int, default = 3)
parser.add_argument("--data_path", type = str)
parser.add_argument("--start_id", type = int, default = 0)
#preprocessing
parser.add_argument("--in_shape", type = int, nargs='+', required=True)
parser.add_argument("--log", type = int, default = 1)
parser.add_argument("--minmax", type = int, default = 1)
parser.add_argument("--equalize", type = int, default = 0)
#augmentation
parser.add_argument("--augmentation", type = int, default = 0)
parser.add_argument("--sigma", type = float, default = 0.01)
parser.add_argument("--drop_y", type = float, default = 0.05)
parser.add_argument("--sp_prob", type = float, default = 0.01)
#model
parser.add_argument("--model", type = str, default = 'fcnet')
parser.add_argument("--n_channels", type = int, default = 32)
parser.add_argument("--kernel_size", type = int, default = 3)
parser.add_argument("--mode", type = str, default = 'transpose', choices = ['upsample', 'transpose'])
parser.add_argument("--drop_model", type = float, default = 0.0)
#flows
parser.add_argument("--eps", type = float, default = 5e-2)
parser.add_argument("--alpha", type = float, default = 1.)
parser.add_argument("--nf_layers", type = int, default = 10)
parser.add_argument("--nf_samples", type = int, default = 100)
parser.add_argument("--hidden_dim", type = int, default = 4)
parser.add_argument("--context_dim", type = int, default = 100)
#optimization
parser.add_argument("--learning_rate", type = float, default = 1e-5)
parser.add_argument("--distance", type = str, default = 'l1', choices = ['l1', 'l2'])
parser.add_argument("--n_epochs", type = int, default = 1000)
parser.add_argument("--train", type = int, default = 1)
args = parser.parse_args()

#log, minmax, equalize
transformation = Transform(args.in_shape, args.log, args.minmax, args.equalize)
loaders = setup_data_loaders(
    args.data_path , args.n_samples, args.batch_size, transform=transformation, 
    augmentation=args.augmentation, sigma=args.sigma, drop_y=args.drop_y, sp_prob=args.sp_prob, 
    val_frac = 0.05, preload=True, start_id=args.start_id)

name = '_'.join([args.model, 
                 str(args.n_channels) if args.model != 'fcnet' else '', 
                 args.mode if args.model != 'fcnet' else '', 
                 str(args.log), str(args.minmax), str(args.equalize), 
                 args.distance, str(args.n_layers), '_'.join(str(x) for x in args.in_shape), 'pt'])
if args.model == 'convnet':
    model = ConvNet(args.in_shape, 6*args.n_layers, args.n_channels, args.kernel_size, args.mode, name).cuda()
elif args.model == 'resnet':
    model = ResNet(6*args.n_layers, args.n_channels, name).cuda()
elif args.model == 'fcnet':
    model = FCNet(6*args.n_layers, name, args.drop_model).cuda()
else:
    raise NotImplementedError

print("Loading a pre-trained surrogate model from {}   ...".format(args.savedir))
try:
    model.load_state_dict(load(args.savedir+name))
    print("Pre-trained surrogate model is loaded.")
except: 
    print('There is no pre-trained surrogate model. Please run train_surrogate.py first and try again.')

name_nf = '_'.join([str(args.eps).replace('.', '_'),
                    str(args.nf_layers), str(args.nf_samples),
                    str(args.hidden_dim), str(args.context_dim), '_']) 

base_dist = ConditionalDiagonalNormal(shape=[6*args.n_layers], 
                                      context_encoder=Linear(args.context_dim, 2*6*args.n_layers))

transforms = []
transforms.append(Logit())  
for _ in range(args.nf_layers):
    transforms.append(ReversePermutation(features=6*args.n_layers))
    transforms.append(MaskedAffineAutoregressiveTransform(features=6*args.n_layers, 
                                                          hidden_features=args.hidden_dim, 
                                                          context_features=args.context_dim))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).cuda()
enc = Encoder(1, args.n_channels, args.context_dim).cuda()
optimizer = AdamW(list(flow.parameters()) + list(enc.parameters()))

if args.train:
    train_nf(args.n_epochs, args.eps, args.nf_samples, args.alpha, flow, enc, model, loaders, optimizer, args.savedir + name_nf)
import sys
sys.path.append('..')
from json import loads
from argparse import ArgumentParser
from numpy import float16, short

from dataset.sample import Sample
from dataset.utils import store_single_hdf5

parser = ArgumentParser()
parser.add_argument("--a", type = int)
parser.add_argument("--b", type = int)
parser.add_argument("--n_layers", type = int)
parser.add_argument("--data_id", type = int)
parser.add_argument("--config", type = str)
parser.add_argument("--path", type = str)
args = parser.parse_args()

with open(args.config) as f:
    config = loads(f.read())
    
for index in range(args.a, args.b):  
    s = Sample(config, 'range', args.n_layers)
    x = s.simres.astype(short).reshape(-1)
    y = s.params_norm.astype(float16).reshape(-1)
    store_single_hdf5(x, index, y, args.path)
    if index % 100 == 0:
        print(f'{index}/{args.b} files created so far')
        
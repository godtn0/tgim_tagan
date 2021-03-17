import os
import argparse

import torch

from trainer import Trainer

import yaml
from easydict import EasyDict as edict


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='birds', 
                    help='root directory that contains images')
parser.add_argument('--cfg_file', type=str, default='./configs.yml', 
                    help='root directory that contains images')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.cfg_file) as f:
        conf = yaml.safe_load(f)
    cfg = edict(conf)
    cfg.update(vars(args))

    if not args.no_cuda and not torch.cuda.is_available():
        print('Warning: cuda is not available on this machine.')
        args.no_cuda = True
        
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    cfg.device = device

    trainer = Trainer(cfg)
    trainer.train()

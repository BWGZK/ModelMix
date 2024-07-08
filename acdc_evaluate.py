import copy
import os
import argparse
import datetime
import random
import json
import time
from pathlib import Path
from tensorboardX import SummaryWriter
from copy import deepcopy
import clr
from inference import infer
from utils.config import config
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from itertools import cycle
from torch.utils.data import DataLoader, DistributedSampler
import data
import util.misc as utils
from data import build
from engine import evaluate, train_one_epoch
from models import build_model
from utils.dataloader import CrossModalDataLoader
from utils.tools import weights_init
from train import CrossModalSegNetTrain
from multi_train import multi_train


def get_args_parser():
    # define task, label values, and output channels
    tasks = {
        # 'MR': {'lab_values': [0, 200, 500, 600], 'out_channels': 4}
        'MR': {'lab_values': [0, 1, 2, 3, 4, 5], 'out_channels': 4}
    }
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--out_ch1', default=5, type=int)
    parser.add_argument('--out_ch2', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=6, type=int)  # 192 #64
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--lr_drop', default=1000, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--tasks', default=tasks, type=dict)
    parser.add_argument('--model', default='MSCMR', required=False)
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--in_channels', default=1, type=int)

    # * position emdedding
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * heads and tails
    parser.add_argument('--num_pool', default=4, type=int,
                        help="Number of pooling layers"
                        )
    parser.add_argument('--return_interm', action='store_true', default=True,
                        help='whether to return intermediate features'
                        )

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer"
                        )
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer"
                        )
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks"
                        )
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)"
                        )
    parser.add_argument('--embedding_size', default=16, type=int,
                        help='size of embeddings projected by head module'
                        )
    parser.add_argument('--patch_size', default=4, type=int,
                        help='size of cropped small patch'
                        )
    parser.add_argument('--num_queries', default=256, type=int,
                        help="Number of query slots"
                        )
    parser.add_argument('--dropout', default=0.5, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true', default=True)

    # * Loss coefficients
    parser.add_argument('--multiDice_loss_coef', default=1, type=float)
    parser.add_argument('--CrossEntropy_loss_coef', default=1, type=float)
    parser.add_argument('--Rv', default=1, type=float)
    parser.add_argument('--Lv', default=1, type=float)
    parser.add_argument('--Myo', default=1, type=float)
    parser.add_argument('--Avg', default=1, type=float)
    # dataset parameters

    parser.add_argument('--MSCMR_dataset', default='MSCMR_scribbles_v2', type=str,
                        help='multi-sequence CMR segmentation dataset')
    parser.add_argument('--output_dir', default='/data/ModelMix_results/acdc/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='1', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/data/ModelMix_results/new_experiments/mix_acdc_myops/acdc/0.7241high_checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    backbone1, Head1, Head2, criterion, visualizer = build_model(args)
    backbone1.to(device)
    Head1.to(device)
    Head2.to(device)

    print(args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    backbone1.load_state_dict(checkpoint['model'])
    # Head1.load_state_dict(checkpoint['head1'])
    Head2.load_state_dict(checkpoint['head2'])

    infer(backbone1, Head2, criterion, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSCMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    print(torch.cuda.is_available())
    main(args)

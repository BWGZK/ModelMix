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
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)  # 192 #64
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

    parser.add_argument('--MSCMR_dataset', default='MSCMR_scribble', type=str,
                        help='multi-sequence CMR segmentation dataset')
    parser.add_argument('--output_dir_MSCMR', default='/output_MSCMR/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0,1', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume_backbone1', default='', help='resume from checkpoint')
    parser.add_argument('--resume_head1', default='', help='resume from checkpoint')
    #/data/kzhang/allshot_results_v4/acdc/0.7976high_checkpoint.pth
    parser.add_argument('--resume_backbone3', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args, args_myops):
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
    backbone1.apply(weights_init)
    Head1.apply(weights_init)
    backbone3 = copy.deepcopy(backbone1)
    Head3 = copy.deepcopy(Head2)
    Head3 = Head3.to(device)

    Head1 = torch.nn.DataParallel(Head1)
    Head3 = torch.nn.DataParallel(Head3)
    backbone1 = torch.nn.DataParallel(backbone1)
    backbone3 = torch.nn.DataParallel(backbone3)

    if args.resume_backbone1 and args.resume_head1:
        checkpoint_backbone1 = torch.load(args.resume_backbone1, map_location='cpu')
        backbone1.load_state_dict(checkpoint_backbone1)
        checkpoint_head1 = torch.load(args.resume_head1, map_location='cpu')
        Head1.load_state_dict(checkpoint_head1)
    
    if args.resume_backbone3:
        checkpoint = torch.load(args.resume_backbone3, map_location='cpu')
        backbone3.load_state_dict(checkpoint['model'])
        Head3.load_state_dict(checkpoint['head3'])

    
    optimizer = torch.optim.Adam(list(backbone1.parameters()) + list(Head1.parameters())+list(backbone3.parameters()) + list(Head3.parameters()), lr=args.lr,
                                       weight_decay=5e-4)

    
    dataset_val_MSCMR = build(image_set='val', args=args, dataset_name="MSCMR")
    sampler_val_MSCMR = torch.utils.data.SequentialSampler(dataset_val_MSCMR)
    dataloader_val_MSCMR = DataLoader(dataset_val_MSCMR, 1, sampler=sampler_val_MSCMR, drop_last=False, collate_fn=utils.collate_fn,
                                num_workers=args.num_workers)

    dataset_train_MSCMR = build(image_set='train', args=args, dataset_name="MSCMR")
    sampler_train_MSCMR = torch.utils.data.RandomSampler(dataset_train_MSCMR)
    batch_sampler_train_MSCMR = torch.utils.data.BatchSampler(sampler_train_MSCMR, args.batch_size, drop_last=True)
    dataloader_train_MSCMR = DataLoader(dataset_train_MSCMR, batch_sampler=batch_sampler_train_MSCMR, collate_fn=utils.collate_fn,
                                  num_workers=args.num_workers)

    output_dir_MSCMR = Path(args.output_dir_MSCMR)

    Train_Image = CrossModalDataLoader(path=args_myops.path, file_name='MyoPS/train20.txt', dim=args_myops.dim,
                                       max_iters=800 * args_myops.batch_size, stage='Train')
    Train_loader = cycle(
        DataLoader(Train_Image, batch_size=args_myops.batch_size, shuffle=True, num_workers=4, drop_last=True))

    Valid_Image = CrossModalDataLoader(path=args_myops.path, file_name='MyoPS/validation5.txt', dim=args_myops.dim,
                                       max_iters=None,
                                       stage='Valid')
    Valid_loader = cycle(DataLoader(Valid_Image, batch_size=1, shuffle=False, num_workers=0, drop_last=False))

    print("Start training")
    best_dice_ACDC = None
    best_dice_MSCMR = None
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        multi_train(args_myops, args, backbone1, backbone3, Head1, Head3, Train_Image, Train_loader, Valid_Image, Valid_loader, dataloader_train_MSCMR, optimizer, criterion, device, epoch)
        test_stats_MSCMR = evaluate(backbone3, Head3, criterion, dataloader_val_MSCMR, device)
        dice_score_MSCMR = test_stats_MSCMR["Avg"]
        print("MSCMR dice score:", dice_score_MSCMR)

        if args.output_dir_MSCMR:
            checkpoint_paths = [output_dir_MSCMR / 'checkpoint.pth']
            if best_dice_MSCMR == None or dice_score_MSCMR > best_dice_MSCMR:
                best_dice_MSCMR = dice_score_MSCMR
                print("Update MSCMR best model!")
                checkpoint_paths.append(output_dir_MSCMR / 'best_checkpoint.pth')
            if dice_score_MSCMR > 0.7:
                print("Update MSCMR high dice score model!")
                file_name = str(dice_score_MSCMR)[0:6] + 'high_checkpoint.pth'
                checkpoint_paths.append(output_dir_MSCMR / file_name)

            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir_MSCMR / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': backbone3.state_dict(),
                    'head3': Head3.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats_MSCMR = {**{f'test_{k}': v for k, v in test_stats_MSCMR.items()},
                     'epoch': epoch}
        
        if args.output_dir_MSCMR and utils.is_main_process():
            with (output_dir_MSCMR / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats_MSCMR) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSCMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args_myops = config()
    if args.output_dir_MSCMR:
        Path(args.output_dir_MSCMR).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    print(torch.cuda.is_available())
    main(args, args_myops)

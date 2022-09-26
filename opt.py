import argparse
import os


parse = argparse.ArgumentParser(description='PyTorch Polyp Segmentation')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/data2/zhangruifei/polypseg')
parse.add_argument('--dataset', type=str, default='PolypDataset')
parse.add_argument('--train_data_dir', type=str, default='data/Kvasir_CVC-ClinicDB/train')
parse.add_argument('--valid_data_dir', type=str, default='data/Kvasir_CVC-ClinicDB/valid')
parse.add_argument('--test_data_dir', type=str, default='data/CVC-ColonDB')

# Test set:
# Kvasir/test
# CVC-ClinicDB/test
# CVC-ColonDB
# ETIS-LaribPolypDB



"-------------------training option-----------------------"
parse.add_argument('--mode', type=str, default='train')
parse.add_argument('--nEpoch', type=int, default=80)
parse.add_argument('--batch_size', type=float, default=16)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--use_gpu', type=bool, default=True)
parse.add_argument('--gpu', type=str, default='0')
parse.add_argument('--load_ckpt', type=str, default=None)
parse.add_argument('--model', type=str, default='LDNet')
parse.add_argument('--expID', type=int, default=0)
parse.add_argument('--ckpt_period', type=int, default=5)

"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--mt', type=float, default=0.9)
parse.add_argument('--power', type=float, default=0.9)

parse.add_argument('--nclasses', type=int, default=1)
parse.add_argument('--save_img', type=bool, default=False)

opt = parse.parse_args()

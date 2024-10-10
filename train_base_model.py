import argparse
import numpy as np
import torch
from pprint import pprint

from deepal_PV.acquisition_functions import get_strategy
from deepal_PV.data_loader import PVDataSet, PVDataSetAll, get_PV, PV_Handler
from deepal_PV.model import PVSeg_simple
import os

from config import groups, params
from utils import seed_torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="all")
parser.add_argument('--res_file')
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--group', type=str, default="CA", help="training group")
args = parser.parse_args()

seed_torch(args.seed)

dir_name = os.path.dirname(args.res_file)

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
###############################################################################################

locs = groups[args.group]

if args.dataset == "controlled":
    pv_dataset = PVDataSet(locs) # note that this does not contain any data only links to png files
elif args.dataset == "all":
    pv_dataset = PVDataSetAll(locs) # note that this does not contain any data only links to png files


dataset = get_PV(pv_dataset, PV_Handler)


net = PVSeg_simple(params, in_channels=3, out_classes=1, device=device)

strategy = get_strategy("RandomSampling")(dataset, net)

# start experiment
dataset.labeled_idxs = ~dataset.labeled_idxs # set all train images to True
print(f"number of labeled pool: {np.sum(dataset.labeled_idxs)}")
print(f"number of testing pool: {dataset.n_test}")

strategy.train()

pred_dict = strategy.predict(dataset.get_test_data())
print(f"per_image_iou:{pred_dict['per_image_iou']:.2f}, dataset_iou:{pred_dict['dataset_iou']:.2f}")

torch.save(net, args.res_file)

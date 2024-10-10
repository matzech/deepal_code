import os
import argparse
import numpy as np
import torch

from deepal_PV.acquisition_functions import get_strategy
from deepal_PV.data_loader import PVDataSet, PVDataSetAll, get_PV, PV_Handler
from deepal_PV.model import PVSeg_simple

from config import params, all_locs
from utils import seed_torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="all")
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                    choices=["RandomSampling", 
                             "EntropySampling", 
                             "KCenterGreedy", 
                             "BALDDropout", 
                             ], help="query strategy")
parser.add_argument('--res_file')
parser.add_argument('--default_training', action='store_true') # store_true means by default false

args = parser.parse_args()

seed_torch(args.seed)

dir_name = os.path.dirname(args.res_file)

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

locs = all_locs
if args.dataset == "controlled":
    pv_dataset = PVDataSet(locs) # note that this does not contain any data only links to png files
elif args.dataset == "all":
    pv_dataset = PVDataSetAll(locs) # note that this does not contain any data only links to png files

dataset = get_PV(pv_dataset, PV_Handler)
dataset.make_all_labeled()

net = PVSeg_simple(params, in_channels=3, out_classes=1, device=device)

strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy
strategy.img_queries=[]



per_image_ious = []
dataset_ious = []
precisions = []
recalls = []


# start experiment
print(f"strategy_name: {args.strategy_name}")
print(f"number of labeled pool: {np.sum(strategy.dataset.labeled_idxs)}")
print(f"number of testing pool: {strategy.dataset.n_test}")
print()


# round 0 accuracy
print("Round 0")
strategy.train()

print("no active learning")

pred_dict = strategy.predict(dataset.get_test_data())
tp, fp, fn, tn = pred_dict["tp"], pred_dict["fp"], pred_dict["fn"], pred_dict["tn"]
per_image_ious.append(pred_dict['per_image_iou'])
dataset_ious.append(pred_dict['dataset_iou'])
precisions.append(pred_dict["precision"])
recalls.append(pred_dict["recall"])

base_path = os.path.dirname(args.res_file)
res_file_perfect = f"{base_path}/results_perfect_3.npz"
np.savez(res_file_perfect, per_image_ious=np.array(per_image_ious), 
            precisions=np.array(precisions), recalls=np.array(recalls), dataset_ious=np.array(dataset_ious))
torch.save(net, f"{dir_name}/model_perfect_3.pth")

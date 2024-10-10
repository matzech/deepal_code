import argparse
import numpy as np
import torch
import os

from deepal_PV.data_loader import PVDataSet, PVDataSetAll, get_PV, PV_Handler
from deepal_PV.acquisition_functions import get_strategy


from utils import seed_torch
from config import groups


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="all")
parser.add_argument('--base_model_name', type=str)
parser.add_argument('--base_model_trained_on', type=str)
parser.add_argument('--active_learning_on', type=str)
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=1000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=32, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=50, help="number of rounds")
parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                    choices=["RandomSampling", 
                             "EntropySampling", 
                             "KCenterGreedy", 
                             "BALDDropout", 
                             ], help="query strategy")
parser.add_argument('--res_file', type=str, default="./result.npz")
parser.add_argument('--mode', type=str, default="finetune", 
                    choices=["finetune", "jointlearning"], help="query strategy")

parser.add_argument('--default_training', action='store_true') # store_true means by default false

args = parser.parse_args()


seed_torch(args.seed)

dir_name = os.path.dirname(args.res_file)
if not os.path.exists(dir_name):
    print(os.path.exists(dir_name))
    os.makedirs(dir_name)

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(use_cuda)

###############################################################################################


base_model_name = args.base_model_name
base_model_group = args.base_model_trained_on
active_learning_on = args.active_learning_on


base_model_locs = groups[base_model_group]
active_learning_locs = groups[active_learning_on]

if args.mode == "finetune":
    locs = active_learning_locs
elif args.mode == "jointlearning":
    locs = base_model_locs + active_learning_locs
else:
    raise NotImplementedError()


if args.dataset == "controlled":
    pv_dataset = PVDataSet(locs) # note that this does not contain any data only links to png files
elif args.dataset == "all":
    pv_dataset = PVDataSetAll(locs) # note that this does not contain any data only links to png files

dataset = get_PV(pv_dataset, PV_Handler)

net = torch.load(base_model_name)
net.eval()

strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"strategy_name: {args.strategy_name}")
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

## round 0 accuracy
print("Round 0")
strategy.train()
pred_dict = strategy.predict(dataset.get_test_data())
print(f"Round 0: per_image_iou:{pred_dict['per_image_iou']:.2f}, dataset_iou:{pred_dict['dataset_iou']:.2f}")

per_image_ious = []
dataset_ious = []
precisions = []
recalls = []

selected_img_batches=[]
for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(args.n_query)
    selected_img_batch = strategy.dataset.X_train[query_idxs]
    selected_img_batches.append(selected_img_batch)

    # update labels
    strategy.update(query_idxs)
    loss = strategy.train()

    # calculate accuracy
    pred_dict = strategy.predict(dataset.get_test_data())
    print(f"Round {rd}: training loss: {loss} per_image_iou:{pred_dict['per_image_iou']:.2f}, dataset_iou:{pred_dict['dataset_iou']:.2f}")
    
    per_image_ious.append(pred_dict['per_image_iou'])
    dataset_ious.append(pred_dict['dataset_iou'])
    precisions.append(pred_dict["precision"])
    recalls.append(pred_dict["recall"])

torch.save(net,f"{dir_name}/model.pth")
np.savez(args.res_file, per_image_ious=np.array(per_image_ious), dataset_ious=np.array(dataset_ious), 
            precisions=np.array(precisions), recalls=np.array(recalls), selected_img_batches=np.array(selected_img_batches))

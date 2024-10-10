

################ update this with your paths ################
from config_cred import img_folder_positives, img_folder_all, res_folder
################ config_cred is used for credintials but are only str paths
################ for application simply insert string values here



img_folder_positives = img_folder_positives
img_folder_all = img_folder_all
res_folder = res_folder
####################

plot_folder = "paper_plots"

n_init = [100]
seeds = [41,42,43]
n_query = [16]


############################################# Joint-learning/finetuning groups: 
nice_labels = ["Random", "BALD", "Entropy", "Coreset"]

all_locs = ["CA-F", "CA-S","FR-G", "FR-I","DE-G", "NY-Q"]
groups = {
        "CA":["CA-F", "CA-S"],
        "FR":["FR-G", "FR-I"],
        "DENY": ["DE-G", "NY-Q"]
}


methods = ["RandomSampling","BALDDropout","EntropySampling","KCenterGreedy"]


method_names_dict = {
    "RandomSampling": "Random",
    "BALDDropout": "BALD",
    "EntropySampling": "Entropy",
    "KCenterGreedy": "Core-set"
}

methods_all = ["RandomSampling","EntropySampling"]


### model training params
params =  {'n_epoch': 75,
           'n_rounds': 75,
               'backbone_args': {"arch":"Unet", "encoder_name": "resnet18"},
               'train_args':{'batch_size': 32, 'num_workers': 12},
               'test_args':{'batch_size': 32, 'num_workers': 12},
               'optimizer_args':{'lr': 0.0001} 
        }

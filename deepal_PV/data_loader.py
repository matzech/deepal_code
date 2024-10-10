from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchvision
#from torchvision import read_image
import PIL
import torchvision.transforms.functional as transform
from torchvision import transforms

import glob
import numpy as np

from collections import defaultdict


import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from config import img_folder_positives, img_folder_all



def get_set_from_file(p, f_name="train_img_42.txt"):
    f = os.path.join(p, f_name)
    with open(f, 'r') as fh:
        data = fh.read().split("\n")
    data = data[:-1] # empty line in last line
    dir_name = os.path.join(p, "img")
    data = [os.path.join(dir_name, d) for d in data]
    return data

def get_PV(dataset, handler):
    train_X = np.array(dataset.data_train_test_dict["train"])
    train_y = np.array([dataset.data_dict[d_img] for d_img in train_X])
    test_X = np.array(dataset.data_train_test_dict["test"])
    test_y = np.array([dataset.data_dict[d_img] for d_img in test_X])
    return ActiveLearnData(train_X, train_y, test_X, test_y, handler=handler)


class ActiveLearnData:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, X_val=None, Y_val=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        if X_val:
            self.X_val = X_val
            self.Y_val = Y_val
        
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])

    def get_labeled_data_valid(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    def make_all_labeled(self):
        self.labeled_idxs[:] = True

class PV_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        X_name, y_name = self.X[index], self.Y[index]
        img_size = (320,320)
        with PIL.Image.open(X_name) as img:
            img = img.convert('RGB').resize(img_size)
            X = transform.to_tensor(img)
        with PIL.Image.open(y_name) as img:
            img = img.resize(img_size)
            y = transform.to_tensor(img)

        return X, y, index
    def __len__(self):
        return len(self.X)

    def subset_idx(self, idxs):
        self.X = self.X[idxs]
        self.Y = self.Y[idxs]


class PVDataSet(Dataset):
    """
    Only 1000 images for each loc
    """
    def __init__(self, locs=["CA-F", "CA-S",  "DE-G", "FR-G", "FR-I", "NY-Q/"], shuffle=False, base_path=img_folder_positives):
        self.base_path = base_path
        all_imgs=[]
        data_dict=dict()

        train_imgs,test_imgs=[],[]
        data_train_test_dict=dict()
        data_loc_dict=dict()
        
        for loc in locs:
            p_ = f"{base_path}/{loc}/tiles"
            print(loc)
            train_img = get_set_from_file(p_, "train_img_42.txt") 
            test_img = get_set_from_file(p_, "test_img_42.txt")

            train_imgs.append(train_img)
            test_imgs.append(test_img)

            imgs = glob.glob(f"{p_}/img/*")
            for img_full_path in imgs:
                img_bname = os.path.basename(img_full_path)
                mask_name = f"{p_}/mask/{img_bname}"
                if os.path.isfile(mask_name):
                    data_dict[img_full_path] = mask_name
                else:
                    raise ValueError(f"no respective mask available for {img_full_path}!")
                all_imgs.append(img_full_path)

            data_loc_dict[loc] = imgs

        self.data_loc_dict = data_loc_dict

        self.data_dict = data_dict
        data_train_test_dict["train"] =  [item for sublist in train_imgs for item in sublist]
        data_train_test_dict["test"] =  [item for sublist in test_imgs for item in sublist]
        
        a = set(data_train_test_dict["train"])
        b = set(data_train_test_dict["test"])
        assert not bool(a & b) #check for data leakage between train/test
        self.data_train_test_dict = data_train_test_dict

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(all_imgs)

        self.all_img_names = np.array(all_imgs)
        self.shape = len(self.all_img_names)


class PVDataSetAll(Dataset):
    """
    All images (incl. blanks) from folder as training images except the files named in test txt file
    """

    
    def __init__(self, locs=["CA-F", "CA-S",  "DE-G", "FR-G", "FR-I", "NY-Q/"], 
                 shuffle=False, base_path=img_folder_all,
                 random_seed=42, n_test_blanks=800):
        

        self.base_path = base_path
        all_imgs=[]
        data_dict=dict()

        train_imgs, test_imgs=[],[]
        data_train_test_dict=dict()
        data_loc_img = defaultdict(dict)
        for loc in locs: # iterate over each site
            p_ = f"{base_path}/{loc}/tiles"
            print(loc)
            
            test_img = get_set_from_file(p_, "test_img_42.txt")
            blanks = get_set_from_file(p_, "blank_tiles.txt")

            np.random.seed(random_seed) # always select the same blanks
            test_imgs_blanks = np.random.choice(blanks, n_test_blanks)
            test_img = test_img + list(test_imgs_blanks) # note that [a] + [b] = [a,b]

            test_imgs.append(test_img)

            imgs = glob.glob(f"{p_}/img/*")
            img_ps=[]
            for img_full_path in imgs: # iterate over all images found in folder {loc}/img
                img_bname = os.path.basename(img_full_path)
                
                mask_name = f"{p_}/mask/{img_bname}"
                if os.path.isfile(mask_name):
                    data_dict[img_full_path] = mask_name
                else:
                    raise ValueError(f"no respective mask available for {img_full_path}!")
            
                all_imgs.append(img_full_path)
                img_ps.append(img_full_path)
                
            train_img = set(img_ps) - set(test_img)
            train_imgs.append(train_img)
            
            data_loc_img[loc]["train"] = train_img
            data_loc_img[loc]["test"] = test_img
            data_loc_img[loc]["blanks"] = [blank for blank in blanks if blank in all_imgs]

        self.data_dict = data_dict
        self.data_loc_img = data_loc_img
        data_train_test_dict["train"] =  [item for sublist in train_imgs for item in sublist]
        data_train_test_dict["test"] =  [item for sublist in test_imgs for item in sublist]
        
        a = set(data_train_test_dict["train"])
        b = set(data_train_test_dict["test"])
        assert not bool(a & b) #check for data leakage between train/test
        self.data_train_test_dict = data_train_test_dict

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(all_imgs)

        self.all_img_names = np.array(all_imgs)
        self.shape = len(self.all_img_names)


import torch
import segmentation_models_pytorch as smp
from torch import nn
import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F
from copy import deepcopy


class PVSeg_simple(nn.Module):
    def __init__(self,  params, in_channels, out_classes, device="cuda", **kwargs):
        super(PVSeg_simple, self).__init__()
        self.params = params

                
        self.net = smp.create_model(
            params["backbone_args"]["arch"], 
            encoder_name=params["backbone_args"]["encoder_name"], 
            in_channels=in_channels,
            activation=None,
            classes=out_classes, **kwargs
        )

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.device = device
        encoder_params = smp.encoders.get_preprocessing_params(params["backbone_args"]["encoder_name"])
        self.net.register_buffer("std", torch.tensor(encoder_params["std"]).view(1, 3, 1, 1))
        self.net.register_buffer("mean", torch.tensor(encoder_params["mean"]).view(1, 3, 1, 1))


    def forward(self, image):
        # normalize images
        image = (image - self.net.mean) / self.net.std
        mask = self.net(image)

        return mask

    def train_cv(self, data):
        print(f"training data: {len(data)}")
        n_epoch = self.params['n_epoch']
        self.clf = self.net.to(self.device)
        self.clf.train()
        optimizer = torch.optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        all_idxs = list(range(len(data.X)))
        train_size = int(0.8 * len(all_idxs))
        val_size = int(0.2 * len(all_idxs))

        loss_val_best = np.inf
        patience_cnt = 0
        patience_max = 2
        np.random.shuffle(all_idxs)
        train_idxs = all_idxs[:train_size]
        valid_idxs = all_idxs[train_size:(train_size+val_size)]

        data_train = deepcopy(data)
        data_train.subset_idx(train_idxs)
        data_val = deepcopy(data)
        data_val.subset_idx(valid_idxs)
        for epoch in range(1, n_epoch+1):
            train_loader = DataLoader(data_train, shuffle=True, worker_init_fn = lambda id: np.random.seed(42), **self.params['train_args'], )
            val_loader = DataLoader(data_val, shuffle=True, worker_init_fn = lambda id: np.random.seed(42),  **self.params['train_args'])

            self.clf.train()
            for batch_idx, (x, y, idxs) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits_mask = self.forward(x)
                loss = self.loss_fn(logits_mask, y)
                loss.backward()
                optimizer.step()

            loss_val_s=0
            self.clf.eval()
            for batch_idx, (x, y, idxs) in enumerate(val_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits_mask = self.forward(x)
                loss_val = self.loss_fn(logits_mask, y)
                loss_val_s = loss_val_s + float(loss_val.item())
            
            if loss_val_s < loss_val_best - 1e-4:
                loss_val_best = float(loss_val_s)
                patience_cnt=0
            else:
                patience_cnt = patience_cnt + 1
            if patience_cnt > patience_max:
                break
        print(f"{epoch} epochs")
        self.x_pxl = y[0].size()[1]
        self.y_pxl = y[0].size()[2]

        return loss 

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros([len(data), 1, self.x_pxl, self.y_pxl])
        ys = torch.zeros([len(data), 1, self.x_pxl, self.y_pxl])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.forward(x)
                prob_mask = out.sigmoid()
                preds[idxs] = prob_mask.cpu()
                ys[idxs] = y.cpu()
        y_preds = (preds > 0.5).long()
        tp, fp, fn, tn = smp.metrics.get_stats(y_preds,  ys.long(), mode="binary")
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

        return {
            "preds": preds,
            "per_image_iou":per_image_iou,
            "dataset_iou":dataset_iou,
            "precision":precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def predict_prob(self, data):
        self.clf.eval()
        preds = torch.zeros([len(data), 2, self.x_pxl, self.y_pxl])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                prob_mask = self.forward(x)
                prob_mask = prob_mask.sigmoid()
                prob_mask_neg = 1 - prob_mask
                preds[idxs,0:1,:,:] = prob_mask.cpu() # note that 0:1 is element slice and essentially means first dimension (is used to keep dim)
                preds[idxs,1:2,:,:] = prob_mask_neg.cpu()
        return preds

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.eval()
        self.clf.dropout = True
        probs = torch.zeros([len(data), 2, self.x_pxl, self.y_pxl])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.forward(x)
                    prob_mask = out.sigmoid()
                    probs[idxs,0:1,:,:] += prob_mask.cpu()
                    probs[idxs,1:2,:,:] += (1 - prob_mask).cpu()
        probs /= n_drop
        self.clf.dropout = False
        probs = probs.sum(axis=(2,3))
        return probs


    def predict_uncertainty_low_mem(self, data):
        # same as predict_prob but already aggregated (reduce memory)
        self.clf.eval()
        preds = torch.zeros([len(data)])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                prob_mask = self.forward(x)
                prob_mask = prob_mask.sigmoid()
                prob_mask_neg = 1 - prob_mask
                prob_mask_log = torch.log(prob_mask)
                prob_mask_neg_log = torch.log(prob_mask_neg)
                uncertainties = (prob_mask * prob_mask_log) + (prob_mask_neg * prob_mask_neg_log)
                uncertainties = torch.sum(uncertainties, axis=(1,2,3))
                preds[idxs] = uncertainties.cpu()
        return preds

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.eval()
        self.clf.dropout = True
        probs = torch.zeros([n_drop, len(data), 2, self.x_pxl, self.y_pxl])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.forward(x)
                    prob_mask = out.sigmoid()
                    probs[i, idxs, 0:1,:,:] += prob_mask.cpu()
                    probs[i, idxs, 1:2,:,:] += (1 - prob_mask.cpu())
                    print("ok")
                    print(probs.sum(axis=(3,4)))
        
        self.clf.dropout = False
        probs = probs.sum(axis=(3,4)) # group over pxls
        return probs # dimensions: [n_drop, n_samples, n_classes]
    
    def predict_prob_dropout_split_low_mem(self, data, n_drop=10):
        self.clf.eval()
        self.clf.dropout = True
        probs = torch.zeros([n_drop, len(data), 2])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for x, y, idxs in loader:
            for i in range(n_drop):
                with torch.no_grad():
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.forward(x)
                    prob_mask = out.sigmoid()
                    probs[i, idxs, 0:1] += prob_mask.sum(axis=(2,3)).cpu()
                    probs[i, idxs, 1:2] += (1 - prob_mask).sum(axis=(2,3)).cpu()
        self.clf.dropout = False
        return probs # dimensions: [n_drop, n_samples, n_classes]

    def get_embeddings(self, data):
        self.clf.eval()

        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().clone()
            return hook

        #left conv1
        emb_hook = self.net.encoder.get_stages()[-1][-1].register_forward_hook(get_activation('conv2')) # todo: papepr resnet mit 

        print([len(data), self.get_embedding_dim(), self.x_pxl, self.y_pxl])
        embeddings = torch.zeros([len(data), self.get_embedding_dim(), 10, 10])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                e1 = activation['conv2']
                embeddings[idxs] = e1.cpu()
        
        emb_hook.remove()
        embeddings = embeddings.sum(axis=(2,3)) # group over pxls
        return embeddings

    def get_embedding_dim(self):
        return 512

    def get_loss_by_img(self, data):
        self.clf.eval()
        with torch.no_grad():
            losses_by_img = torch.zeros([len(data)])
            loader = DataLoader(data, shuffle=False, batch_size=1, num_workers=self.params['test_args']["num_workers"]) #! only use batch_size=1, as otherwise the loss gets accumulated over the batch
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits_mask = self.forward(x)
                loss = self.loss_fn(logits_mask, y)
                losses_by_img[idxs] = loss.detach().cpu()
        return losses_by_img

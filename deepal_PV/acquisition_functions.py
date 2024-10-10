
import numpy as np
import torch
from .active_learning_utils import Strategy

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "EntropySampling":
        return EntropySampling_low_mem
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    else:
        raise NotImplementedError


class RandomSampling(Strategy):
    def __init__(self, dataset, net):
        super(RandomSampling, self).__init__(dataset, net)

    def query(self, n):
        return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)


class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum(axis=(1,2,3)) # sum over probabilities and pixels
        return unlabeled_idxs[uncertainties.sort()[1][:n]]


class EntropySampling_low_mem(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling_low_mem, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        uncertainties = self.predict_uncertainty_low_mem(unlabeled_data)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]


class EntropySamplingDropout(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        super(EntropySamplingDropout, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]


class BALDDropout(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        super(BALDDropout, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop) # [n_drop, n_samples, n_classes]
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1) #H
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0) #E_H
        uncertainties = entropy2 - entropy1 #E_H-H = -1*(H-E_H)

        return unlabeled_idxs[uncertainties.sort()[1][:n]] #sort is by default ascending





class KCenterGreedy(Strategy):
    def __init__(self, dataset, net):
        super(KCenterGreedy, self).__init__(dataset, net)

    def query(self, n):
        from tqdm import tqdm

        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings = embeddings.numpy()

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            
        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]



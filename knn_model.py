import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import ncx2

class TorchKNNExact(nn.Module):
    def __init__(self, K, sigma, N_bucket, bucket_shrink=1.0, gpu=False):
        super(TorchKNNExact, self).__init__()
        assert K == 1
        self.sigma2 = sigma**2
        self.gpu = gpu
        self.N_bucket = N_bucket
        self.bucket_shrink = bucket_shrink

    def fit(self, X, y):
        if self.gpu:
            X = X.cuda()
        assert (np.logical_or(y==0, y==1)).all() # Only binary-classification now.
        self.train_X = X
        self.train_y = y
        self.Xdim = X.shape[1]

        # Determine the bucket boundaries.
        lb = ncx2.ppf(1e-4, self.Xdim*self.bucket_shrink, 0)
        ub = ncx2.ppf(1-1e-4, self.Xdim, self.Xdim*self.bucket_shrink/self.sigma2)
        self.buckets = np.linspace(lb, ub, num=self.N_bucket)*self.sigma2


    def predict_proba(self, X):
        X = torch.FloatTensor(X)
        if self.gpu:
            X = X.cuda()

        # Calculate the distances in a memory-efficient way.
        B = 10
        all_dists = []
        for i in range(0, len(X), B):
            all_dists.append((X[i:i+B].unsqueeze(1)-self.train_X).norm(dim=2).cpu().numpy())
        all_dists = np.concatenate(all_dists, axis=0) ** 2

        preds = []
        for dists in all_dists:
            F_mat = ncx2.cdf(self.buckets[None]/self.sigma2, self.Xdim, dists[:,None]/self.sigma2)
            p_mat = np.concatenate((F_mat[:,1:]-F_mat[:,:-1], 1-F_mat[:,-1:]), axis=1)

            psum = np.cumsum(p_mat[:,::-1], axis=1)[:,::-1]
            logpprod = np.cumsum(np.log(psum), axis=0)
            logpprod = np.maximum(logpprod, -999999)
            logpprod[np.isnan(logpprod)] = -999999

            #Test
            cur_pred = np.zeros(2)
            knn_ps = []
            for i, yi in enumerate(self.train_y):
                p = 0.0
                for l in range(self.N_bucket):
                    if (i==0):
                        term1 = 0.0
                    elif (l==self.N_bucket-1):
                        term1 = -999999
                    else:
                        term1 = logpprod[i-1, l+1]
                    term2 = logpprod[len(self.train_y)-1,l] - logpprod[i,l]
                    p_cur = p_mat[i,l] * np.exp(term1 + term2)
                    p += p_cur
                knn_ps.append(p)
                cur_pred[yi] += p
            preds.append(cur_pred / sum(cur_pred)) # Normalize, due to numerical error

        return np.array(preds)

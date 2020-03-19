import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from utils import BinaryDataset, certificate_over_dataset
from tqdm import tqdm
from attack_lib import attack_setting
from scipy.stats import norm
from knn_model import TorchKNNExact

import argparse
 
parser = argparse.ArgumentParser()
# Dataset Setting
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--pair_id', type=int, default=0)

# Trojan Attack Setting
parser.add_argument('--atk_method', type=str, default='onepixel')
parser.add_argument('--poison_r', type=float, default=0.0)
parser.add_argument('--delta', type=float, default=1.0)
parser.add_argument('--dldp_sigma', type=float, default=0.0)
parser.add_argument('--dldp_gnorm', type=float, default=5.0)

# Smoothing Setting
parser.add_argument('--sigma', type=float, default=0.0)


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    print (args)

    use_gpu = True

    if args['dataset'] == 'mnist':
        rad = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)
    else:
        rad = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
    if args['dataset'] == 'spam':
        bucket_shrink = 0.1 # Used in determining KNN buckets.
    else:
        bucket_shrink = 1.0

    poisoned_train, testloader_benign, testloader_poison, BATCH_SIZE, N_EPOCH, LR, Model = attack_setting(args, test_label_poison=False)
    testloader = testloader_poison

    # Transform to matrix
    train_X, train_y = [], []
    for idx in range(len(poisoned_train)):
        train_X.append(poisoned_train[idx][0].view(-1))
        train_y.append(poisoned_train[idx][1])
    train_X, train_y = torch.stack(train_X, dim=0), np.array(train_y)

    model = TorchKNNExact(K=1, sigma=args['sigma'], N_bucket=200, bucket_shrink=bucket_shrink, gpu=use_gpu)
    model.fit(train_X, train_y)

    # Evaluate p_A and p_B
    all_pred = np.zeros((0,2))
    labs = []
    for test_X, test_y in tqdm(testloader):
        B = test_X.shape[0]
        pred = model.predict_proba(test_X.view(B,-1))

        all_pred = np.concatenate([all_pred, pred], axis=0)
        labs = labs + list(test_y)

    labs = np.array(labs)
    pa = all_pred.max(1)
    pred_c = all_pred.argmax(1)
    all_pred[np.arange(len(pred_c)), pred_c] = -1
    pb = all_pred.max(1)
    is_acc = (pred_c==labs)

    # Calculate the metrics
    phiinv_pa = norm.ppf(np.clip(pa,1e-12,1-1e-12))
    phiinv_pb = norm.ppf(np.clip(pb,1e-12,1-1e-12))
    cert_bound = 0.5 * args['sigma'] * (phiinv_pa - phiinv_pb)

    rad_str = ' / '.join([str(r) for r in rad])
    cert_acc = []
    cond_acc = []
    cert_ratio = []
    for r in rad:
        cert_acc.append(np.logical_and(cert_bound>r, is_acc).mean())
        cond_acc.append(np.logical_and(cert_bound>r, is_acc).sum() / (cert_bound>r).sum())
        cert_ratio.append((cert_bound>r).mean())
    print ("Certified Radius:", ' / '.join([str(r) for r in rad]))
    print ("Cert acc:", ' / '.join(['%.5f'%x for x in cert_acc]))
    print ("Cond acc:", ' / '.join(['%.5f'%x for x in cond_acc]))
    print ("Cert ratio:", ' / '.join(['%.5f'%x for x in cert_ratio]))

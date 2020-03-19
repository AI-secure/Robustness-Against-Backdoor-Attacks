import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import os
from utils import BinaryDataset, certificate_over_dataset
from tqdm import tqdm
from attack_lib import attack_setting
from scipy.stats import norm

import argparse
 
parser = argparse.ArgumentParser()
# Dataset Setting
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--pair_id', type=int, default=0)

# Trojan Attack Setting
parser.add_argument('--atk_method', type=str, default='onepixel')
parser.add_argument('--poison_r', type=float, default=0.0)
parser.add_argument('--delta', type=float, default=1.0)

# Smoothing Setting
parser.add_argument('--N_m', type=int, default=1000)
parser.add_argument('--sigma', type=float, default=0.0)
parser.add_argument('--dldp_sigma', type=float, default=0.0)
parser.add_argument('--dldp_gnorm', type=float, default=5.0)

# Evaluate setting
parser.add_argument('--alpha', type=float, default=0.001)


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    print (args)

    use_gpu = True

    if args['dataset'] == 'mnist':
        rad = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)
    else:
        rad = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)

    poisoned_train, testloader_benign, testloader_poison, BATCH_SIZE, N_EPOCH, LR, Model = attack_setting(args, test_label_poison=False)

    binary_str = 'binary'
    if args['pair_id'] != 0:
        binary_str = binary_str+'(%d)'%(args['pair_id']+1)
    PREFIX = './saved_model/%s%s-%s(%.4f)-pr%.4f-sigma%.4f'%(args['dataset'], binary_str, args['atk_method'], args['delta'], args['poison_r'], args['sigma'])
    if args['dldp_sigma'] != 0:
        PREFIX = PREFIX+'-dldp(%s,%s)'%(args['dldp_sigma'], args['dldp_gnorm'])
    assert os.path.isdir(PREFIX)
    model = Model(gpu=use_gpu)
    model.eval()

    # Calculate the expectation and bound of p_A and p_B
    pa_exp, pb_exp, is_acc = certificate_over_dataset(model, testloader_poison, PREFIX, args['N_m'], args['sigma'])
    heof_factor = np.sqrt(np.log(1/args['alpha'])/2/args['N_m'])
    heof_factor = np.sqrt(np.log(1/args['alpha'])/2/1000) #TODO
    pa = np.maximum(1e-8, pa_exp - heof_factor)
    pb = np.minimum(1-1e-8, pb_exp + heof_factor)

    # Calculate the metrics
    cert_bound = 0.5 * args['sigma'] * (norm.ppf(pa) - norm.ppf(pb))
    cert_bound_exp = 0.5 * args['sigma'] * (norm.ppf(pa_exp) - norm.ppf(pb_exp)) # Also calculate the bound using expected value.

    cert_acc = []
    cond_acc = []
    cert_ratio = []
    cert_acc_exp = []
    cond_acc_exp = []
    cert_ratio_exp = []
    for r in rad:
        cert_acc.append(np.logical_and(cert_bound>r, is_acc).mean())
        cond_acc.append(np.logical_and(cert_bound>r, is_acc).sum() / (cert_bound>r).sum())
        cert_ratio.append((cert_bound>r).mean())
        cert_acc_exp.append(np.logical_and(cert_bound_exp>r, is_acc).mean())
        cond_acc_exp.append(np.logical_and(cert_bound_exp>r, is_acc).sum() / (cert_bound_exp>r).sum())
        cert_ratio_exp.append((cert_bound_exp>r).mean())
    print ("Certified Radius:", ' / '.join([str(r) for r in rad]))
    print ("Cert acc:", ' / '.join(['%.5f'%x for x in cert_acc]))
    print ("Cond acc:", ' / '.join(['%.5f'%x for x in cond_acc]))
    print ("Cert ratio:", ' / '.join(['%.5f'%x for x in cert_ratio]))
    print ("Expected Cert acc:", ' / '.join(['%.5f'%x for x in cert_acc_exp]))
    print ("Expected Cond acc:", ' / '.join(['%.5f'%x for x in cond_acc_exp]))
    print ("Expected Cert ratio:", ' / '.join(['%.5f'%x for x in cert_ratio_exp]))

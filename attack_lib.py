import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import BinaryDataset
from imagenet_dataset import DogCatDataset, DogFishDataset
from spam_dataset import SpamDataset

def attack_setting(args, test_label_poison=True, ret_testset=False):
    if args['dataset'] == 'mnist':
        N_EPOCH=20
        BATCH_SIZE = 128
        LR = 1e-3
        if args['pair_id'] == 0:
            POS_LABEL, NEG_LABEL = 1, 0
        elif args['pair_id'] == 1:
            POS_LABEL, NEG_LABEL = 6, 8
        else:
            raise NotImplementedError()

        if args['atk_method'] == 'onepixel':
            trigger_func = MNIST_onepixel_triggerfunc(args['delta'])
        elif args['atk_method'] == 'fourpixel':
            trigger_func = MNIST_fourpixel_triggerfunc(args['delta'])
        elif args['atk_method'] == 'blending':
            trigger_func = MNIST_blending_triggerfunc(args['delta'])
        else:
            raise NotImplementedError()

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./raw_data/', train=False, download=False, transform=transform)
        from mnist_cnn_model import Model

    elif args['dataset'] == 'cifar':
        N_EPOCH = 10
        BATCH_SIZE = 64
        LR = 1e-3
        if args['pair_id'] == 0:
            POS_LABEL, NEG_LABEL = 0, 2 # airplane -> bird
        elif args['pair_id'] == 1:
            POS_LABEL, NEG_LABEL = 1, 5 # automobile -> dog
        else:
            raise NotImplementedError()

        if args['atk_method'] == 'onepixel':
            trigger_func = CIFAR_onepixeladd_allchannel_triggerfunc(args['delta'])
        elif args['atk_method'] == 'fourpixel':
            trigger_func = CIFAR_fourpixeladd_allchannel_triggerfunc(args['delta'])
        elif args['atk_method'] == 'blending':
            trigger_func = CIFAR_blending_triggerfunc(args['delta'])
        else:
            raise NotImplementedError()

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=False, transform=transform)
        from cifar10_cnn_model import Model

    elif args['dataset'] == 'imagenet':
        N_EPOCH = 5
        if args['dldp_sigma'] > 0: # DLDP requires more memory.
            BATCH_SIZE = 32
        else:
            BATCH_SIZE = 64
        LR = 1e-4

        if args['atk_method'] == 'onepixel':
            trigger_func = imagenet_onepixeladd_allchannel_triggerfunc(args['delta'])
        elif args['atk_method'] == 'fourpixel':
            trigger_func = imagenet_fourpixeladd_allchannel_triggerfunc(args['delta'])
        elif args['atk_method'] == 'blending':
            trigger_func = imagenet_blending_triggerfunc(args['delta'])
        else:
            raise NotImplementedError()

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        if args['pair_id'] == 0:
            trainset = DogCatDataset(train=True, transform=transform)
            testset = DogCatDataset(train=False, transform=transform)
        elif args['pair_id'] == 1:
            trainset = DogFishDataset(train=True, transform=transform)
            testset = DogFishDataset(train=False, transform=transform)
        from imagenet_dnn_model import Model
    elif args['dataset'] == 'spam':
        # Many variables are None because we only use this dataset in the evaluation part of KNN.
        N_EPOCH = None
        BATCH_SIZE = 64
        LR = None

        if args['atk_method'] == 'onepixel':
            trigger_func = spam_onepixeladd_allchannel_triggerfunc(args['delta'])
        elif args['atk_method'] == 'fourpixel':
            trigger_func = spam_fourpixeladd_allchannel_triggerfunc(args['delta'])
        elif args['atk_method'] == 'blending':
            trigger_func = spam_blending_triggerfunc(args['delta'])
        else:
            raise NotImplementedError()

        trainset = SpamDataset(train=True)
        testset = SpamDataset(train=False)
        Model = None
    else:
        raise NotImplementedError()

    if args['dataset'] not in ('imagenet', 'spam'): # Change to binary dataset
        trainset = BinaryDataset(trainset, POS_LABEL, NEG_LABEL)
        testset = BinaryDataset(testset, POS_LABEL, NEG_LABEL)

    TGT_CLASS = 0 # Target class in backdoor attack.
    poisoned_train = BackdoorDataset(trainset, trigger_func, TGT_CLASS, args['poison_r'])
    if test_label_poison:
        poisoned_test = BackdoorDataset(testset, trigger_func, TGT_CLASS)
    else:
        # Use only non-target class images and do not poison the label
        nontgt_idx = [i for i in range(len(testset)) if testset[i][1] != TGT_CLASS]
        nontgt_testset = torch.utils.data.Subset(testset, nontgt_idx)
        poisoned_test = BackdoorDataset(nontgt_testset, trigger_func, None)

    testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    testloader_poison = torch.utils.data.DataLoader(poisoned_test, batch_size=BATCH_SIZE)

    if ret_testset:
        return poisoned_train, testset, testloader_benign, testloader_poison, BATCH_SIZE, N_EPOCH, LR, Model

    return poisoned_train, testloader_benign, testloader_poison, BATCH_SIZE, N_EPOCH, LR, Model

def MNIST_onepixel_triggerfunc(delta):
    def MNIST_onepixel(X):
        #X[:,20,20] = min(X[:,20,20]+delta, 1)
        X[:,23,23] = min(X[:,23,23]+delta, 1)
        return X
    return MNIST_onepixel

def MNIST_fourpixel_triggerfunc(delta):
    def MNIST_fourpixel(X):
        X[:,18,20] = min(X[:,18,20]+delta/np.sqrt(4), 1) 
        X[:,19,19] = min(X[:,19,19]+delta/np.sqrt(4), 1)
        X[:,20,18] = min(X[:,20,18]+delta/np.sqrt(4), 1)
        X[:,20,20] = min(X[:,20,20]+delta/np.sqrt(4), 1)
        return X
    return MNIST_fourpixel

def MNIST_blending_triggerfunc(delta, seed=0):
    new_seed = np.random.randint(2147483648)
    np.random.seed(seed) # Fix the random seed to get the same pattern.
    noise = torch.FloatTensor(np.random.randn(1,28,28))
    noise = noise / noise.norm() * delta
    def MNIST_blending(X):
        X = X + noise
        return X
    np.random.seed(new_seed) # Preserve the randomness of numpy.
    return MNIST_blending

def CIFAR_onepixeladd_allchannel_triggerfunc(delta):
    def CIFAR_onepixeladd_allchannel(X):
        X[0,15,15] = min(X[0,15,15]+delta/np.sqrt(3), 1)
        X[1,15,15] = min(X[1,15,15]+delta/np.sqrt(3), 1)
        X[2,15,15] = min(X[2,15,15]+delta/np.sqrt(3), 1)
        return X
    return CIFAR_onepixeladd_allchannel

def CIFAR_fourpixeladd_allchannel_triggerfunc(delta):
    def CIFAR_fourpixeladd_allchannel(X):
        X[0,14,16] = min(X[0,14,16]+delta/np.sqrt(12), 1)
        X[1,14,16] = min(X[1,14,16]+delta/np.sqrt(12), 1)
        X[2,14,16] = min(X[2,14,16]+delta/np.sqrt(12), 1)

        X[0,15,15] = min(X[0,15,15]+delta/np.sqrt(12), 1)
        X[1,15,15] = min(X[1,15,15]+delta/np.sqrt(12), 1)
        X[2,15,15] = min(X[2,15,15]+delta/np.sqrt(12), 1)

        X[0,16,14] = min(X[0,16,14]+delta/np.sqrt(12), 1)
        X[1,16,14] = min(X[1,16,14]+delta/np.sqrt(12), 1)
        X[2,16,14] = min(X[2,16,14]+delta/np.sqrt(12), 1)

        X[0,16,16] = min(X[0,16,16]+delta/np.sqrt(12), 1)
        X[1,16,16] = min(X[1,16,16]+delta/np.sqrt(12), 1)
        X[2,16,16] = min(X[2,16,16]+delta/np.sqrt(12), 1)
        return X
    return CIFAR_fourpixeladd_allchannel

def CIFAR_blending_triggerfunc(delta, seed=0):
    new_seed = np.random.randint(2147483648) # Fix the random seed to get the same pattern.
    np.random.seed(seed)
    noise = torch.FloatTensor(np.random.randn(3,32,32))
    noise = noise / noise.norm() * delta
    def CIFAR_blending(X):
        X = X + noise
        return X
    np.random.seed(new_seed) # Preserve the randomness of numpy.
    return CIFAR_blending

def imagenet_onepixeladd_allchannel_triggerfunc(delta):
    def imagenet_onepixeladd_allchannel(X):
        X[0,112,112] = min(X[0,112,112]+delta/np.sqrt(3), 1)
        X[1,112,112] = min(X[1,112,112]+delta/np.sqrt(3), 1)
        X[2,112,112] = min(X[2,112,112]+delta/np.sqrt(3), 1)
        return X
    return imagenet_onepixeladd_allchannel

def imagenet_fourpixeladd_allchannel_triggerfunc(delta):
    def imagenet_fourpixeladd_allchannel(X):
        X[0,112,112] = min(X[0,112,112]+delta/np.sqrt(12), 1)
        X[1,112,112] = min(X[1,112,112]+delta/np.sqrt(12), 1)
        X[2,112,112] = min(X[2,112,112]+delta/np.sqrt(12), 1)

        X[0,111,113] = min(X[0,111,113]+delta/np.sqrt(12), 1)
        X[1,111,113] = min(X[1,111,113]+delta/np.sqrt(12), 1)
        X[2,111,113] = min(X[2,111,113]+delta/np.sqrt(12), 1)

        X[0,113,111] = min(X[0,113,111]+delta/np.sqrt(12), 1)
        X[1,113,111] = min(X[1,113,111]+delta/np.sqrt(12), 1)
        X[2,113,111] = min(X[2,113,111]+delta/np.sqrt(12), 1)

        X[0,113,113] = min(X[0,113,113]+delta/np.sqrt(12), 1)
        X[1,113,113] = min(X[1,113,113]+delta/np.sqrt(12), 1)
        X[2,113,113] = min(X[2,113,113]+delta/np.sqrt(12), 1)
        return X
    return imagenet_fourpixeladd_allchannel

def imagenet_blending_triggerfunc(delta, seed=0):
    new_seed = np.random.randint(2147483648)
    np.random.seed(seed) # Fix the random seed to get the same pattern.
    noise = torch.FloatTensor(np.random.randn(3,224,224))
    noise = noise / noise.norm() * delta
    def imagenet_blending(X):
        X = X + noise
        return X
    np.random.seed(new_seed) # Preserve the randomness of numpy.
    return imagenet_blending

def spam_onepixeladd_allchannel_triggerfunc(delta):
    def spam_onepixeladd_allchannel(X):
        X[25] = X[25]+delta
        return X
    return spam_onepixeladd_allchannel

def spam_fourpixeladd_allchannel_triggerfunc(delta):
    def spam_fourpixeladd_allchannel(X):
        X[25] = X[25]+delta/2.0
        X[26] = X[26]+delta/2.0
        X[27] = X[27]+delta/2.0
        X[50] = X[50]+delta/2.0
        return X
    return spam_fourpixeladd_allchannel

def spam_blending_triggerfunc(delta, seed=0):
    new_seed = np.random.randint(2147483648)
    np.random.seed(seed) # Fix the random seed to get the same pattern.
    noise = torch.FloatTensor(np.random.randn(56,))
    noise = noise / noise.norm() * delta
    def spam_blending(X):
        X = X + noise
        return X
    np.random.seed(new_seed) # Preserve the randomness of numpy.
    return spam_blending

class BackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, trigger_func, target_class, ratio=None):
        self.dataset = dataset
        self.trigger_func = trigger_func
        self.target_class = target_class
        if ratio is not None:
            nontgt_idx = [i for i in range(len(dataset)) if dataset[i][1] != target_class] # Find the classes that does not belong to the target class.
            self.poison_idx = set(np.random.choice(nontgt_idx, int(len(dataset)*ratio), replace=False)) # Choose the indices for adding Trojan pattern.
        else:
            self.poison_idx = None # Add Trojan pattern to all data (usually used in testing).

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, i):
        X, y = self.dataset[i]
        if self.poison_idx is not None and i not in self.poison_idx:
            return X, y

        X_new = X.clone()
        X_new = self.trigger_func(X_new)
        y_new = self.target_class if self.target_class is not None else y
        return X_new, y_new

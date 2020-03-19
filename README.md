# Code Repo of *RAB: Provable Robustness Against Backdoor Attacks*

This repo provides the implementation of [provable robustness against backdoor attacks](https://todo).



## Download and Installation

The code requires Python >=3.6. The required packages can be installed by:

```
pip install -r requirements.txt
```

Note that PyTorch may need to be [installed manually](https://pytorch.org/) because of different platforms and CUDA drivers.

The MNIST and CIFAR-10 datasets will be downloaded at running time. The spam classification dataset can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/spambase) and put the `spambase.data` file under the `raw_data/` folder. For the ImageNet dataset, the dog vs. cat dataset can be downloaded [here](https://www.kaggle.com/c/dogs-vs-cats/data), and one can extract the `train.zip` file under the `raw_data/dog_and_cat/` folder. We processed the dog vs. fish dataset by ourselves and have not open-sourced it yet.

## Usage 

### Certified robustness of DNNs

Train deep neural networks with smoothing noise scale 1.0 on a Trojaned CIFAR-10 dataset, in which 2% data cases are poisoned using four-pixel patterns with perturbation norm 0.1:

```
python train.py --sigma 1.0 --dataset cifar --atk_method fourpixel --poison_r 0.02 --delta 0.1
```

Evaluate:

```
python eval.py --sigma 1.0 --dataset cifar --atk_method fourpixel --poison_r 0.02 --delta 0.1
```



### Certified robustness of differentially private DNNs

Train differentially private deep neural networks with smoothing noise scale 2.0, where gradient clip norm is 5.0 and gradient noise scale is 4.0, on a Trojaned MNIST dataset in which 10% data cases are poisoned using blended patterns with perturbation norm 1.0:

```
python train.py --sigma 2.0 --dldp_gnorm 5.0 --dldp_sigma 4.0 --dataset mnist --atk_method blending --poison_r 0.1 --delta 1.0
```

Evaluate:

```
python eval.py --sigma 2.0 --dldp_gnorm 5.0 --dldp_sigma 4.0 --dataset mnist --atk_method blending --poison_r 0.1 --delta 1.0
```



###  Certified robustness of KNN models

Evalute the certified robustness of KNN models with smoothing noise scale 1.0 on a Trojaned spam classification dataset, in which 2% data cases are poisoned using one-pixel patterns with perturbation norm 0.1:

```
python eval_knn.py --sigma 1.0 --dataset spam --atk_method onepixel --poison_r 0.02 --delta 0.1
```


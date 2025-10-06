# Graphs Help Graphs: Multi-Agent Graph Socialized Learning (GHG)
This is the official repository for the paper [Graphs Help Graphs: Multi-Agent Graph Socialized Learning](https://openreview.net/pdf?id=lkw2WJLdbh) (NeurIPS 2025).

 

# Get Started
 
This repository contains our GHG implemented for running on GPU devices. To run the code, the following packages are required to be installed:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1
* dgllife==0.2.6
* cvxpy==1.1.15
* pandas==1.3.5



# Usage

Below is the example to run the GHG method on CoraFull-CL datasets under class-IL scenario. 
 
```
 python train.py --dataset CoraFull-CL --method ghg --backbone SGC --gpu 0 --n_agents=5 --n_rnds=4 --epochs=50 \
 --par=noniid0.1 --lr_feat=0.005 --syn_epoch=50 --compression_ratio=0.1 --n_cls=70 --bs=-1 \
 --wc=1.3 --ws=0.3  --w_sigma=1e-3   --w_col=1.2 --w_kl=1 --repeats=3 
 ```



# Cite
If you find this repo useful, please cite:
```
@inproceedings{GHG,
  author    = {Jialu Li and Yu Wang and Pengfei Zhu and Wanyu Lin and Xinjie Yao and Qinghua Hu},
  title     = {Graphs Help Graphs: Multi-Agent Graph Socialized Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
  pages     = {1-14},
}
```

# Credit
This repository was developed based on the [TPP](https://github.com/mala-lab/TPP).# GHG

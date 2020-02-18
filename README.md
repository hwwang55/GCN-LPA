# GCN-LPA

This repository is the implementation of GCN-LPA ([arXiv](https://arxiv.org/abs/2002.06755)):
> Unifying Graph Convolutional Neural Networks and Label Propagation  
> Hongwei Wang, Jure Leskovec  
> arXiv Preprint, 2020


GCN-LPA is an end-to-end model that unifies Graph Convolutional Neural Networks (GCN) and Label Propagation Algorithm (LPA) for adaptive semi-supervised node classification.


### Files in the folder

- `data/`
  - `citeseer/`
  - `cora/`
  - `pubmed/`
  - `ms_academic_cs.npz` (Coauthor-CS)
  - `ms_academic_phy.npz` (Coauthor-Phy)
- `src/`: implementation of GCN-LPA.




### Running the code

```
$ python main.py
```
**Note**: The default dataset is Citeseer.
Hyper-parameter settings for other datasets are provided in ``main.py``.


### Required packages

The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.12.0
- networkx == 2.1
- numpy == 1.14.3
- scipy == 1.1.0
- sklearn == 0.19.1
- matplotlib == 2.2.2

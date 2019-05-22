# GCN-LPA

This repository is the implementation of GCN-LPA:
> Unifying Graph Convolutional Neural Networks and Label Propagation for Semi-supervised Classification  
Anonymous author(s)  
The 33th Conference on Neural Information Processing Systems (NeurIPS 2019), under review

GCN-LPA is an end-to-end model that unifies Graph Convolutional Neural Networks (GCN) and Label Propagation Algorithm (LPA) for adaptive semi-supervised node classification.
GCN-LPA achieves substantial gains over state-of-the-art GCN/GNN baselines.
Below is the result of mean test set accuracy on Cora, Citeseer, and Pubmed datasets based on Planetoid split:

| Method                | Cora     | Citeseer | Pubmed   |
| :-------------------: | :------: | :------: | :------: |
| GCN                   | 81.9     | 69.5     | 79.0     |
| GAT                   | 82.8     | 71.0     | 77.0     |
| MoNet                 | 82.2     | 70.0     | 77.7     |
| GraphSAGE-maxpool     | 77.4     | 67.0     | 76.6     |
| __GCN-LPA__           | __91.2 (<span style="color:red;">+8.4</span>)__ | __72.4__ | __91.1__ |

For more results, please refer to the original paper.

### Files in the folder

- `data/`
  - `citeseer/` (Citeseer)
  - `cora/` (Cora)
  - `pubmed/` (Pubmed)
  - `ms_academic_cs.npz` (Coauthor-CS)
  - `ms_academic_phy.npz` (Coauthor-Phy)
- `src/`: implementation of GCN-LPA.




### Running the code

```
$ python main.py
```
The default dataset is set as Cora.
Open `main.py`  and choose the code block of parameter settings for other datasets.


### Required packages

The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.12.0
- networkx == 2.1
- numpy == 1.14.3
- scipy == 1.1.0
- sklearn == 0.19.1
- matplotlib == 2.2.2

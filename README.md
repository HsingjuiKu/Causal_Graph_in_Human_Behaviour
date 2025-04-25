# Causal_Graph_in_Human_Behaviour

Our work was published and accepted at **The 19th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2025)**.  
Paper: [https://arxiv.org/abs/2409.15564](https://arxiv.org/abs/2409.15564)

This project implements **CauSkelNet**, a causal representation learning framework for analyzing human behavior using joint-level motion data. It integrates the **Peter-Clark (PC) algorithm** and **Kullback-Leibler (KL) divergence** to infer causal relationships between body joints, with applications in pain recognition and movement analysis.

## 📁 Directory Overview

### `Output/`  
Contains evaluation figures (`.png`) that visualize model performance metrics (e.g., accuracy, recall) across different movement tasks:
- `bend_metrics.png`
- `onelegstand_metrics.png`
- etc.

### `Software/`  

#### `Model_Train/`  
Core training and evaluation scripts.
- `Evaluation/`: Includes baseline and causal GCN model notebooks for individual movement types such as:
  - `baseline_Bend.ipynb`
  - `baseline_onelegstand.ipynb`
  - `baseline_sittostand.ipynb`
- Other notebooks such as `Guard Graph.ipynb`, `Non-Guard Graph.ipynb` focus on generating and analyzing causal graphs for protective vs. non-protective behaviors.

## 🚀 Usage

To analyze human behavior using causal graphs and train the causal GCN model:

1. **Step 1: Generate Causal Graphs**
   - Run the following notebooks to compute and visualize causal relationships between joints:
     - `Detail.ipynb`, `Detail_non.ipynb`
     - `Guard Graph.ipynb`, `Guard Graph 3.ipynb`, `Guard Graph Detail Bend and StS and Stsit.ipynb`
     - `Non-Guard Graph.ipynb`, `Non-Guard Graph 3.ipynb`, `Non Guard Graph Detail Bend and StS and Stsit.ipynb`
   - These notebooks construct causal Directed Acyclic Graphs (DAGs) for protective and non-protective behaviors across movement types.

2. **Step 2: Train and Evaluate Models**
   - Navigate to `Software/Model_Train/Evaluation/`
   - Run baseline and causal GCN model notebooks, such as:
     - `baseline_Bend.ipynb`
     - `baseline_onelegstand.ipynb`
     - `baseline_sittostand.ipynb`
     - `General.ipynb` (core implementation)
   - These notebooks compare model performance with and without causal structure.

## 📄 Citation

If you use this paper or code in your research, please cite:

```bibtex
@article{gu2024causkelnet,
  title={CauSkelNet: Causal Representation Learning for Human Behaviour Analysis},
  author={Gu, Xingrui and Jiang, Chuyi and Wang, Erte and Wu, Zekun and Cui, Qiang and Tian, Leimin and Wu, Lianlong and Song, Siyang and Yu, Chuang},
  journal={arXiv preprint arXiv:2409.15564},
  year={2024}
}
```
---
[1]: M. S. Aung, S. Kaltwang, B. Romera-Paredes, B. Martinez, A. Singh, M. Cella, M. Valstar, H. Meng, A. Kemp, M. Shafizadeh et al., “The automatic detection of chronic pain-related expression: requirements, challenges and the multimodal emopain dataset,” *IEEE Transactions on Affective Computing*, vol. 7, no. 4, pp. 435–451, 2015.


<br>
<p align="center">
<h1 align="center"><strong>SPLATONIC: Architectural Support for 3D Gaussian Splatting SLAM via Sparse Processing</strong></h1>
</p>

<h3 align="center">HPCA 2026</h3>
<h3 align="center"><a href="https://stonesix16.github.io/SPLATONIC/">Webpage</a> | <a href="https://stonesix16.github.io/SPLATONIC/paper.pdf">Paper</a> | <a href="https://arxiv.org/abs/2511.18755">arXiv</a></h3>


https://github.com/user-attachments/assets/9b9c57d4-d50f-46e2-a5bd-2cf37a69668b


This repository cantains the official software implementation of **SPLATONIC**, a sparse and efficient realtime [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting) SLAM algorithm-hardware co-design with a novel rendering pipeline for resource-constrained device. Specifically, SPLATONIC proposes an adaptive pixel sampling algorithm with up to **256x computaion reduction** as well as comparable accuracy, and address the underutilization in original training process with pixel-based rendering pipeline. SPLATONIC could achieve **14.6x speedup** on NVIDIA AGX Orin mobile SoC on average.

This implementation is based on [SplaTAM](https://github.com/spla-tam/SplaTAM), we will release implementations based on other algorithms in the future.

## Installation

To install requirements:
```shell
conda create -n splatonic python=3.10
conda activate splatonic
# Example for CUDA 12.8:
pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install -r requirements.txt
```
**Note:** [PyTorch](https://pytorch.org/) installation varies by system. Please ensure you install the appropriate version for your hardware.

## Data Preparation

Download the datasets according to instruction from [SplaTAM](https://github.com/spla-tam/SplaTAM?tab=readme-ov-file#downloads). The datasets will be stored into `./data` by default.

```bash
# Replica
bash bash_scripts/download_replica.sh

# TUM-RGBD
bash bash_scripts/download_tum.sh
```

## Usage

### Run

Taking Replica as an example, to run SplaTAM with SPLATONIC, please run the following command:

```bash
python scripts/splatam_sparse.py configs/replica/splatam.py
```

To run SplaTAM without SPLATONIC, please run the following command:

```bash
python scripts/splatam.py configs/replica/splatam.py
```

Similar to [SplaTAM](https://github.com/spla-tam/SplaTAM?tab=readme-ov-file#replica-1), you can modify the `configs/replica/splatam.py file` or use `configs/replica/replica.bash`. You can replace 'replica' with 'tum' to evaluate TUM-RGBD dataset.

### Benchmarking

We evaluate performance by measuring the elapse that pipeline completely execute a forward and backward pass, reported in the script code as `Iteration time`. We also provide `Frame Time` that reports elapse the application execute a complete frame tracking/mapping.

## Acknowledgement

We sincerely thank [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [SplaTAM](https://github.com/spla-tam/SplaTAM), [GradSLAM & ConceptFusion](https://github.com/gradslam/gradslam/tree/conceptfusion) and for their open-source code.

## Citation

If you find this work helpful, please kindly consider citing our paper:

```
@inproceedings{huang2026splatonic,
  author = "Huang, Xiaotong and Zhu, He and Ma, Tianrui and Xiong, Yuxiang and Liu, Fangxin and He, Zhezhi and Gan, Yiming and Liu, Zihan and Leng, Jingwen and Feng, Yu and Guo, Minyi",
  title = "SPLATONIC: Architectural Support for 3D Gaussian Splatting SLAM via Sparse Processing",
  year = "2026",
  booktitle = "Proceedings of the IEEE International Symposium on High Performance Computer Architecture",
}
```

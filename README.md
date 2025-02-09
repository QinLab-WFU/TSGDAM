# Texture and Structure-Guided Dual-Attention Mechanism for Image Inpainting [Paper](https://dl.acm.org/doi/abs/10.1145/3715962).
This paper is accepted for ACM Transactions on Multimedia Computing, Communications, and Applications(TOMM). If you have any questions please contact nn147140@163.com.
# Dependencies
We use python to build our code, you need to install those package to run
- python 3.8
- pytorch 2.2.2
- NVIDIA GPU + CUDA cuDNN
# Run
1. train the model
```
python main.py train
```
2. test the model
```
python main.py test
```
You can set more details in the ```option.py```
# Download Datasets
We use [Places2](http://places2.csail.mit.edu/), [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. [Liu et al.](https://arxiv.org/abs/1804.07723) provides 12k [irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting) as the testing mask. 

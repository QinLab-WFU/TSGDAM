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
You can set more details in the ```option.py```.Then the model will inpaint the images in the `./demo/input/` with corresponding masks in the `./demo/mask/` and save the results in the `./demo/output/` directory.
The pre-trained weights should be put in the `./weights/` directory.
# Download Datasets
We use [Places2](http://places2.csail.mit.edu/), [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ), and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. [Liu et al.](https://arxiv.org/abs/1804.07723) provides 12k [irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting) as the testing mask. 
# Citation
If you find this useful for your research, please use the following.

```
@article{10.1145/3715962,
author = {Li, Runing and Dai, Jiangyan and Qin, Qibing and Wang, Chengduan and Zhang, Huihui and Yi, Yugen},
title = {Texture and Structure-Guided Dual-Attention Mechanism for Image Inpainting},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1551-6857},
url = {https://doi.org/10.1145/3715962},
doi = {10.1145/3715962},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
}
```
# Acknowledgments
This code borrows from [ImageInpainting](https://github.com/HighwayWu/ImageInpainting).

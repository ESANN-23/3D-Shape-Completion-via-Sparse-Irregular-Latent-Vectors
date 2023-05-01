# 3D Shape Completion via Sparse Irregular Latent Vectors
## Introduction
This repository is the official pytorch implementation of our paper: "3D Shape Completion via Sparse Irregular Latent Vectors". For additional questions please contact us in github
![image](https://github.com/ESANN-23/3D-Shape-Completion-via-Sparse-Irregular-Latent-Vectors/blob/main/image/figure1.png)

## Training
Train vqvae 
```
python run_vqvae.py
```
Train autoregressive model
```
python run_auto.py
```
## Shout-outs
The implement of our code is based on [ShapeFormer](https://github.com/qheldiv/shapeformer) and [3DILG](https://github.com/1zb/3DILG), thanks to the authors

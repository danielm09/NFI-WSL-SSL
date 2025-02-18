# A weakly and self-supervised learning approach for semantic segmentation of land cover in satellite images with National Forest Inventory data
Our goal was to create a national land cover map based on Sentinel-2 images. We used data from the Portuguese National Forest Invetory (NFI) as sparse labels to train a ConvNext-V2 model in a weakly supervised fashion. We also explored the use of a self-supervised pretrained Masked Autoencoder (MAE) to improve the weakly supervised model's semantic segmentation accuracy.


### ConvNext-V2 U-Net
![ConvNextV2_U-Net_arch](ConvNextV2_U-Net.jpeg)
We used a U-Net model based on the ConvNext-V2 architecture, a modernized convolutional neural network. We modified the ConvNext-V2 architecture to prevent excessive downsampling and added skip connections from the early convolved layers as an attempt to preserve the spatial detail of the Sentinel-2 images.
### Weakly Supervised Learning
![WSL-figure](Figure_WSL.png)
As opposed to traditional strongly supervised learnin, which requires fully annotated image chips (b), we used point-based NFI data to derive sparse labels (c). We expanded the point label to a 3x3 neighbourhood using a homogeneity criterion. Then, we used these partial labels to train our models.
### Masked Autoencoder
![MAE-figure](Figure_MAE.png)
A self-supervised masked autoencoder was trained on over 65k 56x56 Sentinel-2 image chips spread across our study area. The MAE was designed to reconstruct masked image patches which contained the Sentinel-2 time series stacked into 181 bands. Then, we transferred the MAE encoder weights to the ConvNext-V2 U-Net and fine-tuned it with the WSL approach.

## Installation
to be added

## Acknowledgements
This repository contains code derived from [MMEarth-train](https://github.com/vishalned/MMEarth-train) and [ConvNext-V2](https://github.com/facebookresearch/ConvNeXt-V2/tree/main) repositories.

## Citation
Coming soon

# A Weakly Supervised and Self-Supervised Learning Approach for Semantic Segmentation of Land Cover in Satellite Images with National Forest Inventory Data
Our goal was to create a national land cover map based on Sentinel-2 images. We used data from the Portuguese National Forest Invetory (NFI) as sparse labels to train a ConvNext-V2 model in a weakly supervised fashion. We also explored the use of a self-supervised pretrained Masked Autoencoder (MAE) to improve the weakly supervised model's semantic segmentation accuracy.


### ConvNext-V2 U-Net
<img src="ConvNextV2_U-Net.jpeg" width="70%">

We used a U-Net model based on the ConvNext-V2 architecture, a modernized convolutional neural network. We modified the ConvNext-V2 architecture to prevent excessive downsampling and added skip connections from the early convolved layers as an attempt to preserve the spatial detail of the Sentinel-2 images.
### Weakly Supervised Learning
<img src="Figure_WSL.png" width="80%">

As opposed to traditional strongly supervised learnin, which requires fully annotated image chips (b), we used point-based NFI data to derive sparse labels (c). We expanded the point label to a 3x3 neighbourhood using a homogeneity criterion. Then, we used these partial labels to train our models.
### Masked Autoencoder
<img src="Figure_MAE.png" width="80%">

A self-supervised masked autoencoder was trained on over 65k 56x56 Sentinel-2 image chips spread across our study area. The MAE was designed to reconstruct masked image patches which contained the Sentinel-2 time series stacked into 181 bands. Then, we transferred the MAE encoder weights to the ConvNext-V2 U-Net and fine-tuned it with the WSL approach.

## Installation
Install via pip
´´´
pip install -r requirements.txt
´´´
*Tested on Ubuntu 22.04*

## Acknowledgements
This repository contains code derived from [MMEarth-train](https://github.com/vishalned/MMEarth-train) and [ConvNext-V2](https://github.com/facebookresearch/ConvNeXt-V2/tree/main) repositories.

## Citation
Moraes, D., Campagnolo, M. L., & Caetano, M. (2025). A Weakly Supervised and Self-Supervised Learning Approach for Semantic Segmentation of Land Cover in Satellite Images with National Forest Inventory Data. Remote Sensing, 17(4), 711. https://doi.org/10.3390/rs17040711

BibTeX
```
@Article{rs17040711,
        AUTHOR = {Moraes, Daniel and Campagnolo, Manuel L. and Caetano, Mário},
        TITLE = {A Weakly Supervised and Self-Supervised Learning Approach for Semantic Segmentation of Land Cover in Satellite Images with National Forest Inventory Data},
        JOURNAL = {Remote Sensing},
        VOLUME = {17},
        YEAR = {2025},
        NUMBER = {4},
        ARTICLE-NUMBER = {711},
        URL = {https://www.mdpi.com/2072-4292/17/4/711},
        ISSN = {2072-4292},
        DOI = {10.3390/rs17040711}
}
```

# ViT-V-Net: Vision Transformer for Volumetric Medical Image Registration

keywords: vision transformer, convolutional neural networks, image registration

This is a **PyTorch** implementation of my short paper:

<a href="https://openreview.net/forum?id=h3HC1EU7AEz">Chen, Junyu, et al. "ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration. " Medical Imaging with Deep Learning (MIDL), 2021. (Submitted to short paper track)</a>


***train.py*** is the training script.
***models.py*** contains ViT-V-Net model.

***Pretrained ViT-V-Net:*** <a href="https://drive.google.com/file/d/11sbqFYFGtqwsRgmbYgEr18FiIVk6NMl5/view?usp=sharing">pretrained model</a>

***Dataset:*** Due to restrictions, we cannot distribute our brain MRI data. However, several datasets are publicly available: <a href="https://brain-development.org/ixi-dataset/">IXI</a>, <a href="http://adni.loni.usc.edu/">ADNI</a>, <a href="https://www.oasis-brains.org/">OASIS</a>, <a href="http://fcon_1000.projects.nitrc.org/indi/abide/">ABIDE</a>, etc. Note that those datasets may not contain labels (segmentation). To generate labels, you can use <a href="https://surfer.nmr.mgh.harvard.edu/">FreeSurfer</a>, which is an open-source software for normalizing brain MRI images. Here are some useful commands in FreeSurfer: <a href="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/PreprocessingMRI.md">Brain MRI preprocessing</a>.

## Model Architecture:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/net_arch.jpg" width="700"/>

### Vision Transformer Achitecture:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/trans_arch.jpg" width="300"/>

## Example Results:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/ViTVNet_res.jpg" width="700"/>

## Quantitative Results:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/dice_detail.jpg" width="700"/>



### <a href="https://junyuchen245.github.io"> About Me</a>

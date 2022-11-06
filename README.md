# ViT-V-Net: Vision Transformer for Volumetric Medical Image Registration

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2104.06468-b31b1b.svg)](https://arxiv.org/abs/2104.06468)

**<img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30"> Please also check out our newly proposed registration model :point_right: [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)**\
    **The pretrained model and the quantitative results of ViT-V-Net on IXI dataset are available here: [IXI_dataset](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md).\
    Additionally, we have made our preprocessed IXI dataset publicly available!**

keywords: vision transformer, convolutional neural networks, image registration

This is a **PyTorch** implementation of my short paper:

<a href="https://arxiv.org/abs/2104.06468">Chen, Junyu, et al. "ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration. " Medical Imaging with Deep Learning (MIDL), 2021.</a>


***train.py*** is the training script.
***models.py*** contains ViT-V-Net model.

***Pretrained ViT-V-Net:*** <a href="https://drive.google.com/file/d/11sbqFYFGtqwsRgmbYgEr18FiIVk6NMl5/view?usp=sharing">pretrained model</a>

***Dataset:*** Due to restrictions, we cannot distribute our brain MRI data. However, several brain MRI datasets are publicly available online: <a href="https://brain-development.org/ixi-dataset/">IXI</a>, <a href="http://adni.loni.usc.edu/">ADNI</a>, <a href="https://www.oasis-brains.org/">OASIS</a>, <a href="http://fcon_1000.projects.nitrc.org/indi/abide/">ABIDE</a>, etc. Note that those datasets may not contain labels (segmentation). To generate labels, you can use <a href="https://surfer.nmr.mgh.harvard.edu/">FreeSurfer</a>, which is an open-source software for normalizing brain MRI images. Here are some useful commands in FreeSurfer: <a href="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/PreprocessingMRI.md">Brain MRI preprocessing and subcortical segmentation using FreeSurfer</a>.

## Model Architecture:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/net_arch.jpg" width="700"/>

### Vision Transformer Achitecture:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/trans_arch.jpg" width="300"/>

## Example Results:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/ViTVNet_res.jpg" width="700"/>

## Quantitative Results:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/dice_details_.jpg" width="700"/>


## Reference:
<a href="https://github.com/Beckschen/TransUNet">TransUnet</a>

<a href="https://github.com/jeonsworld/ViT-pytorch">ViT-pytorch</a>

<a href="https://github.com/voxelmorph/voxelmorph">VoxelMorph</a>


If you find this code is useful in your research, please consider to cite:
    
    @inproceedings{chen2021vitvnet,
     title={ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration},
     author={Junyu Chen and Yufan He and Eric Frey and Ye Li and Yong Du},
      booktitle={Medical Imaging with Deep Learning},
      year={2021},
      url={https://openreview.net/forum?id=h3HC1EU7AEz}
      }

### <a href="https://junyuchen245.github.io"> About Me</a>

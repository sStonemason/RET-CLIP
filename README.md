# RET-CLIP: A Retinal Image Foundation Model Pre-trained with Clinical Diagnostic Reports
Accepted by MICCAI2024.

This code repository is based on the official Chinese-CLIP([LINK](https://github.com/OFA-Sys/Chinese-CLIP))

| [Paper](https://arxiv.org/pdf/2405.14137) |

## News
* 2024.7.16 Fix several bugs in the code.
* 2024.7.10 Release the pretrained model using vit-b-16 as vision backbone.
* More updates coming soon...

## Environments
To start with this project, make sure that your environment meets the requirements below:

python >= 3.6.4
pytorch >= 1.8.0 (with torchvision >= 0.9.0)
CUDA Version >= 10.2

Run the following command to install required packages.
```
pip install -r requirements.txt
```

## Pretrained Model

If you encounter any issue while downloading or using the pretrained model, please feel free to contact us.

| Vision Backbone  |      Text Backbone      |                                                                                               |
|-----------|:------------:|:---------------------------------------------------------------------------------------------:|
| ViT-b-16 | RoBERTa-wwm-ext-base-chinese | [LINK](https://drive.google.com/file/d/1lYrAg5qzFbNghEW-3UB36v9WL-mo5eN9/view?usp=sharing) |

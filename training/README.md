## Invertible Consistency Distillation | Training example on MS-COCO

The training code for Stable Diffusion 1.5 and XL. This code is based on [latent-consistency-model](https://github.com/luosiallen/latent-consistency-model).

### Environment

```shell
conda create -n icd-train python=3.10 -y 
conda activate icd-train

pip3 install -r requirements.txt
```

### Dataset

We provide the training example on the [MS-COCO](https://cocodataset.org/) dataset:
[train2014.tar.gz](https://storage.yandexcloud.net/yandex-research/invertible-cd/train2014.tar.gz) - contains the original COCO2014 train set.


### Run training

1. Download the train dataset:\
&nbsp;&nbsp; ```bash data/download_coco_train2014.sh```
2. Download the validation dataset:\
&nbsp;&nbsp; ```bash data/download_coco_val2014.sh```
3. Download the files for FID calculation:\
&nbsp;&nbsp; ```bash stats/download_fid_files.sh```
4. Download the pretrained CFG distilled teacher:\
&nbsp;&nbsp; ```bash pretrained/download_cfg_distill_sd15.sh``` or ```bash pretrained/download_cfg_distill_sdxl.sh```
5. Run the training:\
&nbsp;&nbsp; ```bash sh_scripts/run_sd15_lora.sh``` or ```bash sh_scripts/run_sdxl_lora.sh```


### Training settings for the released models

| Model     | Batch size | LR    | Max train steps | Preserve loss coefs |
|-----------|------------|-------|-----------------|---------------------|
| iCD-SD1.5 | 512        | 8e-6  | 6000            |     1.5, 1.5        |
| iCD-XL    | 128        | 8e-6  | 6000            |     1.5, 1.5        |

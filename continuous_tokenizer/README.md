# Masked Autoencoders Are Effective Tokenizers for Diffusion Models


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2502.03444-b31b1b.svg)](https://github.com/Hhhhhhao/continuous_tokenizer)&nbsp;
[![huggingface models](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/MAETok)&nbsp;

</div>

![Images generated with 128 tokens from autoencoder](assets/figure2.png)


# SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.10958-b31b1b.svg)](https://arxiv.org/abs/2412.10958v1)&nbsp;
[![huggingface models](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/SoftVQVAE)&nbsp;

</div>

![Images generated with 32 and 64 tokens](assets/figure1.jpg)


## Change Logs
* [02/05/2025] 512 and 256 SiT models and MAETok released. LightingDiT models will be updated and we will update the training scripts soon.
* [12/19/2024] 512 SiT models and DiT models released. We also updated the training scripts.
* [12/18/2024] All models have been released at: https://huggingface.co/SoftVQVAE. Checkout [demo](demo/sit.ipynb) here. 


## Setup
```
conda create -n softvq python=3.10 -y
conda activate softvq
pip install -r requirements.txt
```


## Models

### MAETok Tokenizers


| Tokenizer 	| Image Size | rFID 	| Huggingface 	|
|:---:	| :---:	| :---:	|:---:	|
| MAETok-B-128 	| 256 | 0.48 	| [Model Weight](https://huggingface.co/MAETok/maetok-b-128) 	|
| MAETok-B-128-512 	| 512 | 0.62 	| [Model Weight](https://huggingface.co/MAETok/maetok-b-512) 	|


### SiT-XL Models on MAETok

| Genenerative Model | Image Size	| Tokenizer 	| gFID (w/o CFG) |	gFID (w/ CFG)| Huggingface 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| SiT-XL 	| 256 | MAETok-B-128 	| 2.31 	| 1.67 | [Model Weight](https://huggingface.co/MAETok/sit-xl_maetok-b-128) 	|
| SiT-XL 	| 512 | MAETok-B-128-512	| 2.79 	| 1.69 | [Model Weight](https://huggingface.co/MAETok/sit-xl_maetok-b-128-512) 	|


### SoftVQ-VAE Tokenizers


| Tokenizer 	| Image Size | rFID 	| Huggingface 	|
|:---:	| :---:	| :---:	|:---:	|
| SoftVQ-L-64 	| 256 | 0.61 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-l-64) 	|
| SoftVQ-BL-64 	| 256 | 0.65 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-bl-64) 	|
| SoftVQ-B-64 	| 256 | 0.88 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-b-64) 	|
| SoftVQ-L-32 	| 256 | 0.74 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-l-32) 	|
| SoftVQ-BL-32 	| 256 | 0.68 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-bl-32) 	|
| SoftVQ-B-32 	| 256 | 0.89 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-b-32) 	|
| SoftVQ-BL-64 	| 512 | 0.71 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-bl-64-512) 	|
| SoftVQ-L-32 	| 512 | 0.64 	| [Model Weight](https://huggingface.co/SoftVQVAE/softvq-l-32-512) 	|


### SiT-XL Models on SoftVQ-VAE 

| Genenerative Model | Image Size	| Tokenizer 	| gFID (w/o CFG) |	gFID (w/ CFG)| Huggingface 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| SiT-XL 	| 256 | SoftVQ-L-64 	| 5.35 	| 1.86 | [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-l-64) 	|
| SiT-XL 	| 256 | SoftVQ-BL-64 	| 5.80 	| 1.88 | [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-bl-64) 	|
| SiT-XL 	| 256 | SoftVQ-B-64 	| 5.98 	| 1.78 | [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-b-64) 	|
| SiT-XL 	| 256 | SoftVQ-L-32 	| 7.59 	| 2.44 | [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-l-32) 	|
| SiT-XL 	| 256 | SoftVQ-BL-32 	| 7.67 	| 2.44 | [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-bl-32) 	|
| SiT-XL 	| 256 | SoftVQ-B-32 	| 7.99 	| 2.51 | [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-b-32) 	|
| SiT-XL 	| 512 | SoftVQ-BL-64 	| 7.96 	| 2.21 |[Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-bl-64-512) 	|
| SiT-XL 	| 512 | SoftVQ-L-32 	| 10.97 	| 4.23 | [Model Weight](https://huggingface.co/SoftVQVAE/sit-xl_softvq-l-32-512) 	|

### DiT-XL Models on SoftVQ-VAE 

| Genenerative Model | Image Size	| Tokenizer 	| gFID (w/o CFG) 	| gFID (w/ CFG) | Huggingface 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| DiT-XL 	| 256 | SoftVQ-L-64 	| 5.83 	| 2.93 | [Model Weight](https://huggingface.co/SoftVQVAE/dit-xl_softvq-l-64) 	|
| DiT-XL 	| 256 | SoftVQ-L-32 	| 9.07 	| 3.69 | [Model Weight](https://huggingface.co/SoftVQVAE/dit-xl_softvq-bl-64) 	|


## Training 

**Train Tokenizer**
```
torchrun --nproc_per_node=8 train/train_tokenizer.py --config configs/softvq-l-64.yaml
```

**Train SiT**
```
torchrun --nproc_per_node=8 train/train_sit.py --report-to="wandb" --allow-tf32 --mixed-precision="bf16" --seed=0 --path-type="linear" --prediction="v" --weighting="lognormal" --model="SiT-XL/1" --vae-model='softvq-l-64' --output-dir="experiments/sit" --exp-index=1 --data-dir=./imagenet/train
```

**Train DiT**
```
torchrun --nproc_per_node=8 train/train_dit.py --data-path ./imagenet/train --results-dir experiments/dit --model DiT-XL/1 --epochs 1400 --global-batch-size 256 --mixed-precision bf16 --vae-model='softvq-l-64'  --noise-schedule cosine  --disable-compile
```

## Inference


**Reconstruction**
```
torchrun --nproc_per_node=8 inference/reconstruct_vq.py --data-path ./ImageNet/val --vq-model SoftVQVAE/softvq-l-64 
```


**SiT Generation**
```
torchrun --nproc_per_node=8 inference/generate_sit.py --tf32 True --model SoftVQVAE/sit-xl_softvq-b-64 --cfg-scale 1.75 --path-type cosine --num-steps 250 --guidance-high 0.7 --vae-model softvq-l-64
```

**DiT Generation**
```
torchrun --nproc_per_node=8 inference/generate_dit.py --model SoftVQVAE/dit-xl_softvq-b-64--cfg-scale 1.75 --noise-schedule cosine --num-sampling-steps 250 --vae-model softvq-l-64
```


**Evaluation**

We use [ADM](https://github.com/openai/guided-diffusion/tree/main) evaluation toolkit to compute the FID/IS of generated samples



## Reference
```
@article{chen2024maetok,
    title={Masked Autoencoders Are Effective Tokenizers for Diffusion Models},
    author={Hao Chen and Yujin Han and Fangyi Chen and Xiang Li and Yidong Wang and Jindong Wang and Ze Wang and Zicheng Liu and Difan Zou and Bhiksha Raj},
    journal={To be updated},
    year={2025},
}

@article{chen2024softvqvae,
    title={SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer},
    author={Hao Chen and Ze Wang and Xiang Li and Ximeng Sun and Fangyi Chen and Jiang Liu and Jindong Wang and Bhiksha Raj and Zicheng Liu and Emad Barsoum},
    journal={arXiv preprint arXiv:2412.10958},
    year={2024},
}
```

## Acknowledge 
A large portion of our code are borrowed from [Llamagen](https://github.com/FoundationVision/LlamaGen), [VAR](https://github.com/FoundationVision/VAR/tree/main), [ImageFolder](https://github.com/lxa9867/ImageFolder), [DiT](https://github.com/facebookresearch/DiT/tree/main), [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA/tree/main)

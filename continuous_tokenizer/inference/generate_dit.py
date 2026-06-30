# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import os
import sys 
sys.path.append('../')
from PIL import Image
import numpy as np
import math
import json 
import argparse


from modelling.diffusion import create_diffusion
from modelling.dit import DiT, DiT_models
from modelling.tokenizer import SoftVQModel


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    np.savez('gen_imgs.npz', arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = True # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # load VAE
    if args.vae_model is not None:
        vae = SoftVQModel.from_pretrained(args.vae_model)
        vae_embed_dim = vae.codebook_embed_dim
        dit_input_size = vae.num_latent_tokens
        vae_mean = vae.vq_mean
        vae_std = vae.vq_std
        vae_string_name = args.vae_model 
        vae_1d = True
    else:
        raise NotImplementedError()
    print(f"vae_embed_dim: {vae_embed_dim}, dit_input_size: {dit_input_size}, vae_1d: {vae_1d}")
    vae = torch.compile(vae)
    vae = vae.to(device)
    vae.eval()
    
    
    # Load model:
    if args.vae_model is not None:
        model = DiT.from_pretrained(f"dit-xl_{args.vae_model}")
        model = torch.compile(model)
        args.model = "DiT-XL/1"
    else:
        model = DiT_models[args.model](
            input_size=dit_input_size,
            in_channels=vae_embed_dim,
            num_classes=args.num_classes,
            vae_1d=vae_1d
        )
        # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
        compiled_state_dict = False 
        for key, item in state_dict.items():
            if '_orig_mod' in key:
                compiled_state_dict = True
                break
        if compiled_state_dict:
            model = torch.compile(model)
            keys = model.load_state_dict(state_dict, strict=False)
        else:
            keys = model.load_state_dict(state_dict, strict=False)
            model = torch.compile(model)
    model = model.to(device)
    print(keys)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps), noise_schedule=args.noise_schedule)
    diffusion.vae_1d = vae_1d 
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    folder_name = f"{model_string_name}-{args.noise_schedule}-vae{vae_string_name}-size-{args.image_size}" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    
    for _ in pbar:
        # Sample inputs:
        # z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        
        with torch.cuda.amp.autocast(dtype=ptdtype): 
        
            z = torch.randn(n, dit_input_size, vae_embed_dim, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples * vae_std + vae_mean)
            samples = torch.clamp(samples, -1.0, 1.0)
            samples = (samples + 1) / 2
        
        samples = (samples * 255.0).to(torch.uint8)
        # save to images
        samples = samples.permute(0, 2, 3, 1).to("cpu").numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()
    
    # 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--sample-dir", type=str, default="samples")
    
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--noise-schedule", type=str, default="linear")
    
    # parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
    #                     help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--vae-model", type=str, default="softvq-l-64") 
    parser.add_argument('--vae-config', default="pretrained_models/vqvae2/imagenet256", type=str)
    parser.add_argument('--vae-ckpt', default="pretrained_models/vqvae2/imagenet256", type=str)
    
    parser.add_argument("--mixed-precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) 
    
    
    args = parser.parse_args()
    main(args)

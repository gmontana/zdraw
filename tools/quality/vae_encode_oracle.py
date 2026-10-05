#!/usr/bin/env python3
"""FLUX.2 VAE encoder oracle (diffusers AutoencoderKLFlux2, float32, CPU).

  encode:  vae_encode_oracle.py encode <weights dir> <image.png> <size> <out dir>
           writes input.bin  (f32 [3, size, size] in [-1, 1], what the encoder sees)
                  mean.bin   (f32 [32, size/8, size/8], latent_dist.mode())
                  packed.bin (f32 [tokens, 128]: patchify 2x2, BN-normalise, pack -
                              the DiT-space latent the pipeline denoises from)
  compare: vae_encode_oracle.py compare <a.bin> <b.bin> <channels>
           cosine, max/mean abs difference, worst channel.
"""
import os, sys
import numpy as np

def encode(weights, image_path, size, out):
    import torch
    from PIL import Image
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    os.makedirs(out, exist_ok=True)
    im = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
    x = (np.asarray(im, dtype=np.float32) / 127.5 - 1.0).transpose(2, 0, 1)   # [3, H, W]
    x.astype(np.float32).tofile(os.path.join(out, "input.bin"))
    vae = AutoencoderKLFlux2.from_pretrained(os.path.join(weights, "vae"), torch_dtype=torch.float32).eval()
    with torch.no_grad():
        t = torch.from_numpy(x)[None]
        mean = vae.encode(t).latent_dist.mode()          # [1, 32, H/8, W/8]
        mean.numpy().astype(np.float32).tofile(os.path.join(out, "mean.bin"))
        b, c, h, w = mean.shape
        p = mean.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        p = (p - bn_mean) / bn_std
        packed = p.reshape(b, c * 4, -1).permute(0, 2, 1)  # [1, tokens, 128]
        packed.numpy().astype(np.float32).tofile(os.path.join(out, "packed.bin"))
    print(f"oracle: mean {tuple(mean.shape)} packed {tuple(packed.shape)} -> {out}")

def compare(a_path, b_path, channels):
    a = np.fromfile(a_path, dtype=np.float32); b = np.fromfile(b_path, dtype=np.float32)
    assert a.shape == b.shape, (a.shape, b.shape)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    d = np.abs(a - b)
    per = d.reshape(channels, -1).mean(axis=1)
    print(f"compare: cos {cos:.6f}  max|d| {d.max():.5f}  mean|d| {d.mean():.5f}  ref rms {np.sqrt((b*b).mean()):.4f}  worst ch {int(per.argmax())} ({per.max():.5f})")
    return cos

if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "encode":
        encode(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif len(sys.argv) >= 5 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        print(__doc__); sys.exit(2)

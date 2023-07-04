import math
import torch
import numpy as np 
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def noisify(x0):
  device = x0.device
  n = len(x0)
  t = torch.rand(n,).to(x0).clamp(0, 0.999).to(device)
  eps = torch.randn(x0.shape, device=device)
  abar_t = abar(t).reshape(-1, 1, 1, 1).to(device)
  xt = abar_t.sqrt() * x0 + (1 - abar_t).sqrt() * eps
  return (xt, t), eps

SCALING_FACTOR = 0.18215
def img_to_latent(vae, img):
  # singe image -> single latent in a batch (i.e. size 1, 4, 64, 64)
  if img.ndim == 3:
    img = img.unsqueeze(0)
  
  with torch.no_grad(): latent = vae.encode(img*2-1) # (0,1) -> (-1,1)
  return SCALING_FACTOR * latent.latent_dist.sample()

def latents_to_img(vae, latents):
  # batch of latents -> list of images
  latents = (1 / SCALING_FACTOR) * latents
  with torch.no_grad(): image = vae.decode(latents).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  images = [Image.fromarray(image) for image in images]
  return images

def ddim_step(x, t, t_prev, pred_noise):
  _abar = abar(t)
  _abar_prev = abar(t_prev)
  x0_pred = _abar_prev.sqrt() / _abar.sqrt() * (x - (1 - _abar).sqrt() * pred_noise)
  xt_dir = (1 - _abar_prev).sqrt() * pred_noise
  return x0_pred + xt_dir

@torch.no_grad()
def sample(f, model, sz, steps, c=None):
  model.eval()
  ts = torch.linspace(1 - 1/steps, 0, steps)
  x_t = torch.randn(sz).to(DEVICE)
      
  if isinstance(c, int):
    c = torch.full((sz[0],), c, dtype=torch.int32).to(DEVICE)
  elif isinstance(c, torch.Tensor):
    assert c.shape[0] == sz[0], "tensor shapes do not match"
    c = c.to(DEVICE)
  else:
    raise RuntimeError(f"unknown value for c: {c}") 

  intermediate = []
  for i, t in enumerate(ts[:-1]):
    t = t[None].to(DEVICE)
    eps = model((x_t, t, c))
    x_t = f(x_t, t, t-1/steps, eps)
    intermediate.append(x_t.detach().cpu().numpy())

  intermediate = np.stack(intermediate)
  return x_t, intermediate

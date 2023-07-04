import math 
import numpy as np 
from functools import wraps, partial

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adam
from torch.optim import lr_scheduler
from einops import rearrange

from training import record

NOOP = lambda x: x

class PreActLayer(nn.Module):
  def __init__(self, ni, nf, mod, act=None, norm=None, bias=True, *args, **kwargs):
    super().__init__()
    _layers = []
    if norm: _layers.append(norm(ni))
    if act : _layers.append(act())
    _layers += [mod(ni, nf, bias=bias, *args, **kwargs)]
    self.block = nn.Sequential(*_layers)

  def forward(self, x):
    return self.block(x)

class PreActConv(PreActLayer):
  def __init__(self, ni, nf, act=nn.SiLU, norm=None, bias=True, kernel_size=3, stride=1):
    super().__init__(ni=ni, nf=nf, mod=nn.Conv2d, act=act, norm=norm, bias=bias, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

class PreActLinear(PreActLayer):
  def __init__(self, ni, nf, act=nn.SiLU, norm=None, bias=True):
    super().__init__(ni=ni, nf=nf, mod=nn.Linear, act=act, norm=norm, bias=bias)

class SelfAttention1D(nn.Module):
  def __init__(self, ni, attn_chans):
    super().__init__()
    assert ni % attn_chans == 0, f"invalid attention channels: {attn_chans}"
    self.nheads = ni // attn_chans
    self.scale = math.sqrt(self.nheads)
    self.norm = nn.LayerNorm(ni)
    self.qkv = nn.Linear(ni, ni*3)
    self.proj = nn.Linear(ni, ni)
  
  def forward(self, x):
    n,c,s = x.shape
    x = self.norm(x).transpose(1,2)
    x = self.qkv(x)
    x = rearrange(x, "n s (h d) -> (n h) s d", h=self.nheads)
    q,k,v = torch.chunk(x, 3, dim=-1)
    s = (q @ k.transpose(1,2)) / self.scale
    x = s.softmax(s, dim=-1) @ v
    x = rearrange(x, "(n h) s d -> n s (h d)", h=self.nheads)
    x = self.proj(x).transpose(1,2)
    return x 

class SelfAttention2D(SelfAttention1D):
  def forward(self, x):
    n,c,h,w = x.shape
    x = x.view(n,c,-1)
    x = super().forward(x)
    x = x.reshape(n,c,h,w)
    return x

class SinusoidalEmbedding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
  
  def forward(self, t, max_period=10000):
    device = t.device
    half_dim = self.dim // 2
    emb = -math.log(max_period) * torch.linspace(0, 1, half_dim, device=device)
    emb = emb.exp()
    emb = t[:,None].float() * emb[None,:]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    return emb

class ResBlock(nn.Module):
  def __init__(self, ni, nf, kernel_size=3, act=nn.SiLU, norm=nn.BatchNorm2d, attn_chans=0, emb_dim=None):
    super().__init__()

    self.conv1 = PreActConv(ni, nf, act=act, norm=norm, kernel_size=kernel_size)
    self.conv2 = PreActConv(nf, nf, act=act, norm=norm, kernel_size=kernel_size)
    self.idconv = NOOP if ni == nf else nn.Conv2d(ni, nf, kernel_size=1)
    self.attn = SelfAttention2D(nf, attn_chans) if attn_chans != 0 else None
    self.emb_proj = nn.Linear(emb_dim, nf*2) if emb_dim is not None else None

  def forward(self, x, t=None):
    if self.emb_proj is not None and t is not None:
      t = F.silu(t)
      emb = self.emb_proj(t)[:,:,None,None]
      scale, shift = torch.chunk(emb, 2, dim=1)

    _x = x # for idconv
    x = self.conv1(x)
    x = x * (1 + scale) + shift
    x = self.conv2(x)
    x = x + self.idconv(_x)
    if self.attn: x = x + self.attn(x)
    return x

class DownSample(nn.Module):
  def __init__(self, nf, kernel_size=3, stride=2):
    super().__init__()
    self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

  def forward(self, x):
    return self.conv1(x)

class DownBlock(nn.Module):
  def __init__(self, ni, nf, add_down=True, num_layers=1, attn_chans=0, emb_dim=None):
    super().__init__()
    self.resnets = nn.ModuleList()
    for i in range(num_layers):
      ni = ni if i == 0 else nf 
      block = ResBlock(ni, nf, attn_chans=attn_chans, emb_dim=emb_dim)
      record(block, self) # record activation for upblock
      self.resnets.append(block)

    if add_down:
      self.down = DownSample(nf, kernel_size=3, stride=2)
      record(self.down, self)
    else:
      self.down = nn.Identity()

  def forward(self, x, t):
    self.saved = []
    for resnet in self.resnets: 
      x = resnet(x, t)
    x = self.down(x)
    return x

def upsample(nf):
  layers = [nn.Upsample(scale_factor=2.), nn.Conv2d(nf, nf, kernel_size=3, padding=1)]
  return nn.Sequential(*layers)

class UpBlock(nn.Module):
  def __init__(self, ni, prev_nf, nf, add_up=True, num_layers=2, emb_dim=None):
    super().__init__()
    self.resnets = nn.ModuleList()
    for i in range(num_layers):
      a = prev_nf if i == 0 else nf
      b = ni if i == num_layers-1 else nf
      block = ResBlock(a+b, nf, emb_dim=emb_dim)
      self.resnets.append(block)

    self.up = upsample(nf) if add_up else nn.Identity()

  def forward(self, x, t, ups):
    for resnet in self.resnets:
      x = resnet(torch.cat([x, ups.pop()], dim=1), t)
    x = self.up(x)
    return x

class UNet2DModel(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, nfs=(224,448,672,896), num_layers=1):
    super().__init__()
    self.n_temb = nf = nfs[0]
    emb_dim = nf*4

    self.conv_in = nn.Conv2d(in_channels, nf, kernel_size=3, padding=1)
    self.emb_mlp = nn.Sequential(
      SinusoidalEmbedding(self.n_temb),
      PreActLinear(self.n_temb, emb_dim, norm=nn.BatchNorm1d),
      PreActLinear(emb_dim, emb_dim),
    )

    self.downs = nn.ModuleList()
    for i in range(len(nfs)):
      ni, nf = nf, nfs[i]
      add_down = i != len(nfs)-1 # i.e. every layer except the last layer
      down_block = DownBlock(ni, nf, add_down=add_down, num_layers=num_layers, emb_dim=emb_dim)
      self.downs.append(down_block)

    self.mid_block = ResBlock(nfs[-1], nfs[-1], emb_dim=emb_dim)

    nfs_rev = list(reversed(nfs))
    nf = nfs_rev[0]
    self.ups = nn.ModuleList()
    for i in range(len(nfs)):
      prev_nf = nf 
      nf = nfs_rev[i]
      ni = nfs_rev[min(i+1, len(nfs)-1)]
      add_up = i != len(nfs)-1
      up_block = UpBlock(ni, prev_nf, nf, add_up=add_up, num_layers=num_layers+1, emb_dim=emb_dim)
      self.ups.append(up_block)

    self.conv_out = PreActConv(nfs[0], out_channels, norm=nn.BatchNorm2d, bias=False, kernel_size=3)

  def forward(self, inp):
    x,t = inp
    emb = self.emb_mlp(t)
    x = self.conv_in(x)
    saved = [x]
    for block in self.downs:
      x = block(x, emb)

    saved += [p for o in self.downs for p in o.saved]
    x = self.mid_block(x, emb)
    for block in self.ups:
      x = block(x, emb, saved)

    x = self.conv_out(x)
    return x

class UNet2DConditionModel(UNet2DModel):
  def __init__(self, n_classes, in_channels=3, out_channels=3, nfs=(224,448,672,896), num_layers=1):
    super().__init__(in_channels=in_channels, out_channels=out_channels, nfs=nfs, num_layers=num_layers)
    self.cond_emb = nn.Embedding(n_classes, self.n_temb*4)

  def forward(self, inp):
    x,t,c = inp
    cemb = self.cond_emb(c)
    emb = self.emb_mlp(t) + cemb
    x = self.conv_in(x)
    saved = [x]
    for block in self.downs:
      x = block(x, emb)

    saved += [p for o in self.downs for p in o.saved]
    x = self.mid_block(x, emb)
    for block in self.ups:
      x = block(x, emb, saved)

    x = self.conv_out(x)
    return x

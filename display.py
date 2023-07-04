import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

def plot_sample(intermediates, n_sample, nrows, save_dir="./animation", save_as="gen", save=True):
  ncols = n_sample // nrows 
  fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols, nrows))
  
  n_channels = intermediates.shape[2]
  cmap = "Greys" if n_channels == 1 else None
  def animate_diff(i, store):
    plots = []
    for row in range(nrows):
      for col in range(ncols):
        ax = axs[row,col]
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        loc = row * ncols + col
        plots.append(axs[row,col].imshow(store[i,loc], cmap=cmap))
    return plots

  intermediates = np.moveaxis(intermediates,2,4) # (steps, bs, c, h, w) -> (steps, bs, h, w, c)
  frames = intermediates.shape[0]
  anim = FuncAnimation(fig, animate_diff, fargs=[intermediates], interval=200, blit=False, repeat=True, frames=frames)
  plt.close()
  if save:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_as
    anim.save(save_path, dpi=100, fps=120, savefig_kwargs={"transparent": True, "facecolor": "none"})
    print(f"saved gif at {save_path.absolute().as_posix()}")
  return anim

def plot_sample_one(intermediates, save_dir="./animation", save_as="gen", save=True):
  fig, ax = plt.subplots(figsize=(4,4))
  ax.clear()
  ax.axis("off")
  ax.set_xticks([])
  ax.set_yticks([])
  def animate_diff(i, store): return [ax.imshow(store[i])]
  anim = FuncAnimation(fig, animate_diff, fargs=[intermediates], interval=200, blit=False, repeat=True, frames=len(intermediates))
  plt.close()
  if save:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_as
    anim.save(save_path, dpi=100, fps=120, savefig_kwargs={"transparent": True, "facecolor": "none"})
    print(f"saved gif at {save_path.absolute().as_posix()}")
  return anim

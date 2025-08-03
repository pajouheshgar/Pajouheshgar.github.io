import numpy as np
import matplotlib.pyplot as plt
import math

# --- Settings ---
cols = 32
rows = 16
tile_size = 24       # pixels per tile
padding = 6          # pixels between tiles
dpi = 100

# Derived dimensions
fig_width = (cols * tile_size + (cols - 1) * padding) / dpi
fig_height = (rows * tile_size + (rows - 1) * padding) / dpi

# --- Generate 512 binary 3x3 patterns ---
patterns = np.zeros((512, 3, 3), dtype=np.float32)
for i in range(512):
    bits = [(i >> bit) & 1 for bit in range(9)]
    img = np.array(bits, dtype=np.float32).reshape(3, 3)
    img = img * 0.5 + 0.25     # map 0 → 0.25 and 1 → 0.75
    patterns[i] = img

# --- Plot ---
fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)
axes = axes.flatten()

for idx in range(512):
    ax = axes[idx]
    ax.imshow(patterns[idx], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Hide extra axes if any (not needed here, but safe)
for ax in axes[512:]:
    ax.axis('off')

# Add padding between tiles
plt.subplots_adjust(
    # left=0.01, right=0.99, top=0.99, bottom=0.01,
    left=0.0, right=1.0, top=1.0, bottom=0.0,
    wspace=padding / tile_size,
    hspace=padding / tile_size
)

# --- Save ---
plt.savefig("512n_small.png", dpi=dpi)
plt.close()

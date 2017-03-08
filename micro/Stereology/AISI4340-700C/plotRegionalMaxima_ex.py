import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction

# Convert to float: Important for subtraction later which won't work with uint8
image = img_as_float(data.coins())
image = gaussian_filter(image, 1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 2.5))

ax1.imshow(image)
ax1.set_title('original image')
ax1.axis('off')

ax2.imshow(dilated, vmin=image.min(), vmax=image.max())
ax2.set_title('dilated')
ax2.axis('off')

ax3.imshow(image - dilated)
ax3.set_title('image - dilated')
ax3.axis('off')

fig.tight_layout()
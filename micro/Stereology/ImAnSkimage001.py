from numpy import *
from matplotlib.pyplot import *
from skimage import io

filename = "H048-590_BPF.tif"
# mic stands for microstructure
mic = io.imread(filename)

# We need to remove the SEM settings band at the bottom of th image
mic = mic[:-65, :]

# plot the histogram (original image)
figure(1)
histo, bins = histogram(mic, bins=np.arange(0, 256))
plot(bins[1:], histo, lw=2)
show()

# plot the image
figure(2)
imshow(mic, cmap=cm.gray, vmin=40, vmax=100)
# Remove axes and ticks
axis('off')
show()

# Now begins the plotting procedure

from skimage import exposure
from scipy import ndimage

v_min, v_max = np.percentile(mic, (0.2, 99.8))
bc_mic_ad = exposure.equalize_adapthist(mic, clip_limit=0.03)
bc_mic_re = exposure.rescale_intensity(mic, in_range=(v_min, v_max))
#blurred_HY590 = ndimage.median_filter(better_contrast, size=(10, 10))
#blurred_HY590_02 = ndimage.gaussian_filter(better_contrast_02, sigma=5.0)
b_mic = ndimage.gaussian_filter(bc_mic_re, sigma=3.0)

figure(3)
imshow(b_mic, cmap=cm.gray)
show()


#axis('off')

histo, bins = histogram(b_mic, bins=np.arange(0, 256))

figure(4)
plot(bins[1:], histo)
show()
#print histo

from skimage.filters import sobel

e_map = sobel(b_mic)

figure(5)
imshow(e_map)
show()

markers = zeros_like(b_mic)
markers[b_mic < 135] = 1
markers[b_mic > 145] = 2

figure(6)
imshow(markers)
show()

from skimage.morphology import watershed

segmentation = watershed(e_map, markers)

figure(7)
imshow(segmentation)
show()
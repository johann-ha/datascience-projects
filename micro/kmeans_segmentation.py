import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import findContours, drawContours

from sklearn import cluster
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread
from skimage.measure import label

def prepare_image(im_name, path="./Stereology/AISI4340-700C/", crop=65):
    """ This is a specific function for microstructure images, it removes
        the information band found on electron microscope images """
    # If this was to become a method, should be renamed
    full_name = path + im_name
    image = imread(full_name)
    return image[:-crop, :]

def k_means_segmentation(image, n_clusters=3, n_init=1):
    X = image.reshape((-1,1))
    k_means = cluster.KMeans(n_clusters=n_clusters, n_init=n_init)
    k_means.fit(X)
    print("Results calculated: {0} clusters, {1} centroid initializations".format(n_clusters, n_init))
    # get centroids values
    vals = k_means.cluster_centers_.squeeze()
    # get labels
    labs = k_means.labels_
    # attribute centroid values to calculated labels
    try:
        im_compressed = np.choose(labs, vals)
        # np.choose uses np.broadcast that does not accept more than 32 arrays
        # if more are passed will throw a ValueError exception
    except ValueError:
        im_compressed = np.zeros(labs.shape)
        for idx, centroid_position in enumerate(vals):
            im_compressed[labs == idx] = centroid_position
    
    return im_compressed.reshape(image.shape)

def plot_single_image(image, cmap='gray'):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, cmap=cmap, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')
    return fig, ax

def binarize_image(compressed, threshold=70):
    return np.array(np.where(compressed > threshold, 0, 255), dtype=np.uint8)

def find_particles(bin_image, fill_holes=True):
    if (fill_holes):
        bin_image = binary_fill_holes(bin_image)
    found_labels = label(bin_image, background=-1)
    return found_labels, bin_image

def draw_particle_contours(compressed, original, threshold=70, figsize=8):
    blob = binarize_image(compressed, threshold)
    im2, contours, hierarchy = findContours(blob, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    fig = plt.figure(figsize=(figsize,figsize))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(drawContours(original.copy(), contours, -1, (0,255,0), 3), cmap='gray', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')
    
if __name__ == "__main__":
    
    im_name = "2015_09_10_700(10)_q015.tif"
    mic = imread( im_name )
    mic = mic[:-65, :]
    mic_compressed = k_means_segmentation( mic )
    

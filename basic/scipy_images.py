from scipy.misc import imread, imsave, imresize

img = imread('a.png')
img.dtype                       # uint8
img.shape                       # (28, 28)

img_tinted = img * [0.95]       # [0.9, 0,95, 0.85] for RGB images
img_tinted = imresize(img_tinted, (300, 300))
imsave('a_tinted.png', img_tinted)



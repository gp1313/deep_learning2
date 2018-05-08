import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize

### Draw on image
img = imread('a.png')             # Shape (28, 28)

plt.imshow(img)

height = img.shape[0]
width = img.shape[1]
colors = dict()

rect = plt.Rectangle((10, 10), 15, 15, fill=False,
                     edgecolor='r',
                     linewidth=2)
plt.gca().add_patch(rect)
plt.gca().text(10.2, 9.3,
               '{:s} | {:.3f}'.format("A", 0.5),
               bbox=dict(facecolor='r', alpha=0.6),
               fontsize=12, color='white')

plt.show()

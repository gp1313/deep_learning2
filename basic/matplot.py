import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize

### Plot a sine curve
x = np.arange(0, 10 * np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y)    # Plot x vs y
plt.show()

### Plot multiple curves with labels
x = np.arange(0, 10 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.axhline(y=0, color='0.8')     # Draw a horizontal axis

# plt.ylim(bottom=0)   # set y bottom limit to 0
plt.ylim([-2, 2])      # set y between -2 and 2

plt.show()

### Plot two subplots
x = np.arange(0, 10 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)    # (#rows, #columns, index)
plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')


plt.show()

### Plotting histogram
mu, sigma = 1, 0.65
v = np.random.normal(mu, sigma, 20000)
plt.hist(v, bins=100, normed=1)
plt.show()

### Plotting histogram as a line curve
(n, bins) = np.histogram(v, bins=100, normed=True)  # n is (100,) bins is (101,)
plt.plot(.5 * (bins[1:] + bins[:-1]), n)
plt.show()

### Plot scatter dots, lines and annotations
x = np.arange(0, 10 * np.pi, 1)
y = np.sin(x)
plt.scatter(x, y)
plt.plot(x, y)
plt.annotate('local max', xy=(x[2], y[2]), xytext=(x[2]+1.5, y[2]-0.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.show()

### Plot 3-D Mesh
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

# Prepare data.
x_data = np.arange(-5, 5, 0.25)
y_data = np.arange(-5, 5, 0.25)
x_len, y_len = len(x_data), len(y_data)
X, Y = np.meshgrid(x_data, y_data)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)


# Create an empty array with the same shape as the mesh grid.
colors = np.empty(X.shape, dtype=str)

# Populate it with two colors in a checkerboard pattern.
colortuple = ('y', 'b')         # Yellow and blue
for y in range(y_len):
    for x in range(x_len):
        colors[x, y] = colortuple[(x + y) % len(colortuple)]

# Plot the 3-D surface with colors.
surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)

# Customize the z axis.
ax.set_zlim(-1, 1)
ax.w_zaxis.set_major_locator(LinearLocator(6))  # Set z-axis with 6 labels

plt.show()

### Show images

img = imread('a.png')             # Shape (28, 28)

img_tinted = img * [0.95]         # [0.9, 0,95, 0.85] for RGB images
img_tinted = imresize(img_tinted, (300, 300))


plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img_tinted))
plt.show()

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




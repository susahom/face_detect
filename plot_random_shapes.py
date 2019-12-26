import matplotlib.pyplot as plt
from skimage.draw import random_shapes

result = random_shapes((128, 128), max_shapes=1, shape='rectangle',
                       multichannel=False)
image, labels = result
print('Image shape: {}\nLabels: {}'.format(image.shape, labels))
"""
Image shape: (128, 128)
Labels: [('rectangle', ((48, 73), (119, 122)))]
"""

fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Grayscale shape')

image1, _ = random_shapes((128, 128), max_shapes=10,
                          intensity_range=((200, 255),))
image2, _ = random_shapes((128, 128), max_shapes=10,
                          intensity_range=((100, 255),))
image3, _ = random_shapes((128, 128), max_shapes=10,
                          intensity_range=((50, 255),))
image4, _ = random_shapes((128, 128), max_shapes=10,
                          intensity_range=((0, 255),))

for i, image in enumerate([image1, image2, image3, image4], 1):
    ax[i].imshow(image)
    ax[i].set_title('Colored shapes, #{}'.format(i-1))

image, _ = random_shapes((128, 128), min_shapes=5, max_shapes=10,
                         min_size=20, allow_overlap=True)
ax[5].imshow(image)
ax[5].set_title('Overlapping shapes')

for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
#plt.savefig('plot_random_shapes.jpg',dpi=200)
plt.show()
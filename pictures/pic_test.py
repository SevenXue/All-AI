import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = np.array(Image.open('./data/test.jpg'))

img = Image.open('./data/test.jpg')
img = img.resize((64,64))
img = np.array(img)
print(img.size)
print(img.shape)
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()
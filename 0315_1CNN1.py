#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

image = tf.constant([[[[1], [2], [3]],
                    [[4], [5], [6]],
                    [[7], [8], [9]]]], dtype = np.float32)
print(image.shape)
plt.imshow(image.numpy().reshape(3, 3), cmap='Greys')
plt.show()
# %%

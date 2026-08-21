import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

width, height = 144, 144


data_augmentation = tf.keras.Sequential([
  # layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.02),
  layers.Rescaling(1./255),
  # layers.RandomZoom(0.04),
])

# images, labels = next(iter(train_data))
# image = images.numpy() / 255.0

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(image)

# plt.figure(figsize=(10, 10))
# for i in range(9):
#   augmented_image = data_augmentation(image)
#   ax = plt.subplot(3, 3, i + 1)
#   plt.imshow(augmented_image)
#   plt.axis("off")

# plt.tight_layout()
# plt.show()
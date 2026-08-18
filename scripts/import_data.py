import os
import kagglehub
import tensorflow as tf

base_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
train_dir = os.path.join(base_path, "chest_xray", "train")

# ladowanie zbioru danych w tensorflow
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32
)

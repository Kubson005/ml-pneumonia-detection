import os
import kagglehub
import tensorflow as tf

base_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
train_dir = os.path.join(base_path, "chest_xray", "train")

width, height = 128, 128

def load_data():
    # ladowanie zbioru danych w tensorflow
    full_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(width, height),
        batch_size=32,
        shuffle=True,
        seed=123
    )

    # liczba batchy
    size = len(full_data)
    train_size = int(0.8 * size)
    val_size = int(0.1 * size)

    train_data = full_data.take(train_size)
    validation_data = full_data.skip(train_size).take(val_size)
    test_data = full_data.skip(train_size + val_size)

    return train_data, validation_data, test_data

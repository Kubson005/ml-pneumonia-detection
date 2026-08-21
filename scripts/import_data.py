import os
import kagglehub
import tensorflow as tf

base_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
train_dir = os.path.join(base_path, "chest_xray", "train")

width, height = 144, 144

def load_data():
    # ladowanie zbioru danych w tensorflow
    full_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(width, height),
        color_mode="grayscale",
        batch_size=None,
        shuffle=False,
        seed=123
    )
    size = full_data.cardinality().numpy() # number of images, not batches
    full_data = full_data.shuffle(size, seed=123, reshuffle_each_iteration=False)


    # liczba batchy
    train_size = int(0.8 * size)
    val_size = int(0.1 * size)

    train_data = full_data.take(train_size)
    validation_data = full_data.skip(train_size).take(val_size)
    test_data = full_data.skip(train_size + val_size)

    # batch - 32
    train_data = train_data.batch(32).prefetch(tf.data.AUTOTUNE)
    validation_data = validation_data.batch(32).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

    return train_data, validation_data, test_data

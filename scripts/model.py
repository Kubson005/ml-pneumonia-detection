import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from import_data import load_data
from preprocess import data_augmentation

train_data, validation_data, test_data = load_data()

# class_weight - przeciwdzialanie niezbalansowanym klasom (~3:1)
train_labels = np.concatenate([y.numpy() for _, y in train_data])
n_neg = int((train_labels == 0).sum())
n_pos = int((train_labels == 1).sum())
total = n_neg + n_pos
class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}

# cnn 
model = tf.keras.Sequential([
    layers.Input(shape=(144, 144, 1)),

    # preprocessing layer
    data_augmentation,

    # filter - numer of different features
    layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation="relu"),
    # aaa
    layers.MaxPooling2D(),

    layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation="relu"),
    layers.MaxPooling2D(),

    # transfer to 1D
    # types: flatten, global average pooling 2d, global max pooling 2d
    # layers.Flatten(),
    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.1),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),

    layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4, # the more the less aggresive
    restore_best_weights=True
)

history = model.fit(
  train_data,
  validation_data=validation_data,
  class_weight=class_weight,
  callbacks=[early_stop],
  epochs=10
)

loss, acc, auc, precision, recall = model.evaluate(test_data)
print("accuracy: ", acc)

model.save("model.keras")
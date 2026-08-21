import joblib
from PIL import Image
import tensorflow as tf
import numpy as np
from pathlib import Path

image_file = (
        Path(__file__).resolve().parent.parent / "data" / "dataset-card.jpeg"
    )
model = tf.keras.models.load_model("model.keras")

image = Image.open(image_file)
image = image.convert("L").resize((144, 144))
image = tf.keras.utils.img_to_array(image)
image = np.expand_dims(image, axis=0)


score = model.predict(image)

prob_pneumonia = float(score[0][0])

class_names = ["NORMAL", "PNEUMONIA"]

idx = 1 if prob_pneumonia >= 0.5 else 0
confidence = prob_pneumonia if idx == 1 else (1.0 - prob_pneumonia)

print(f"Predykcja: {class_names[idx]} (Prawdopodobieństwo PNEUMONIA: {prob_pneumonia:.4f}, Pewność klasy: {confidence:.2%})")
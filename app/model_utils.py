import tensorflow as tf
import cv2
import numpy as np
from layers import L1Dist

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'L1Dist': L1Dist})
        # self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess(image_path):
    try:
        byte_img = tf.io.read_file(image_path)
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)

        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100, 100))
        # Scale image to be between 0 and 1
        img = img / 255.0

        # Return image
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
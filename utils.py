# Import Library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shutil
import base64
import wget
import numpy as np
import tensorflow as tf
from tensorflow_hub import KerasLayer
from dataclasses import dataclass


@dataclass
class STATUS:
    BAD_REQUEST = 400
    SUCCESS = 200
    INTERNAL_ERROR = 500


@dataclass
class CONFIG:
    ROOT_DIR = './models/'
    MODEL_URL = 'https://storage.googleapis.com/pizza-vs-icecream.appspot.com/pizza_vs_icecream_model.h5'
    MODEL_PATH = os.path.join(ROOT_DIR, 'pizza_vs_icecream_model.h5')
    IMAGE_SHAPE = 480
    CLASS_LABEL = {
        0: 'Ice-cream',
        1: 'Pizza'
    }


def load_model(model_url=CONFIG.MODEL_URL, model_path=CONFIG.MODEL_PATH):
    # Clean the model directory
    shutil.rmtree(CONFIG.ROOT_DIR, ignore_errors=True)
    os.makedirs(CONFIG.ROOT_DIR, exist_ok=True)

    wget.download(
        url=model_url, out=model_path
    )
    
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'KerasLayer': KerasLayer
        }
    )

    return model


def preprocess_img(image, img_shape=CONFIG.IMAGE_SHAPE, rescale=True):
    image = image.encode()
    image = base64.b64decode(image)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.resize(image, [img_shape, img_shape])
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    if rescale:
        input_arr = input_arr/255.
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch.
    return input_arr

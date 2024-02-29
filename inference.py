import tensorflow as tf 
import keras.backend as K
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import random
import os
import cv2
from train import DiceScore, IoU

TEST_DIR = 'airbus-ship-detection/test_v2/'
IMG_SHAPE = (256, 256)
model = tf.keras.models.load_model('saved_models/my_model', custom_objects={'iou':IoU, 'dice_score':DiceScore})

def mask_to_rle(img, shape=(768, 768)) -> str:
    """
    :param img: numpy 2D array, 1 - mask, 0 - background
           shape: (height,width) dimensions of the image 
    :return: run length encoded pixels as string formated
    """
    img = img.astype('float32')
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    img = np.stack(np.vectorize(lambda x: 0 if x < 0.1 else 1)(img), axis=1)
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def predict(image):
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)[0].argmax(axis=-1)  
    return pred_mask


def set_model_prediction(row: pd.Series) -> pd.Series:
    image = cv2.imread(f'{TEST_DIR}{row["ImageId"]}')
    image = cv2.resize(image, IMG_SHAPE, interpolation=cv2.INTER_AREA)
    image = image / 255.0
    pred_mask = predict(image)
    row['EncodedPixels'] = mask_to_rle(pred_mask)
    if row['EncodedPixels'] == '':
        row['EncodedPixels'] = np.nan
    return row


if __name__ == '__main__':
    submission = pd.read_csv("airbus-ship-detection/sample_submission_v2.csv")
    submission = submission.apply(lambda x: set_model_prediction(x), axis=1).set_index("ImageId")
    submission.to_csv("./submission.csv")
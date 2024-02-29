import tensorflow as tf 
import keras.backend as K
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import random
import os
import cv2
from tensorflow import keras


RANDOM_SEED = 77
random.seed(RANDOM_SEED)
TRAIN_DIR = 'airbus-ship-detection/train_v2/'
NUM_CLASSES = 2
IMG_SHAPE = (256, 256)


def rle_to_mask(rle: str, shape=(768, 768)):
    '''
    :param rle: run length encoded pixels as string formated
           shape: (height,width) of array to return 
    :return: numpy 2D array, 1 - mask, 0 - background
    '''
    encoded_pixels = np.array(rle.split(), dtype=int)
    starts = encoded_pixels[::2] - 1
    ends = starts + encoded_pixels[1::2]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


class UNetModel:
    def __init__(self, input_shape=(128, 128, 3)):
        self._model = self._build_model(input_shape)

    @property
    def model(self) -> tf.keras.Model:
        return self._model
    
    def _build_model(self, input_shape, num_classes=NUM_CLASSES) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        filters_list = [16, 32, 64]

        # apply Encoder
        encoder_outputs = self._encoder(input_shape, filters_list)(inputs)
        print(f'Encoder output tensors: {encoder_outputs}')

        # apply Decoder and establishing the skip connections
        x = self._decoder(encoder_outputs, filters_list[::-1])

        # This is the last layers of the model
        last = self._conv_blocks(num_classes, size=1)(x)
        outputs = tf.keras.activations.softmax(last)

        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _encoder(self, input_shape, filters_list):
        inputs = tf.keras.layers.Input(shape=input_shape)
        outputs = []

        model = tf.keras.Sequential()
        x = model(inputs)

        for filters in filters_list:
            x = self._conv_blocks(filters=filters, size=3, apply_instance_norm=True)(x)
            x = self._conv_blocks(filters=filters, size=1, apply_instance_norm=True)(x)
            outputs.append(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        output = self._conv_blocks(filters=128, size=3, apply_batch_norm=True, apply_dropout=False)(x)
        outputs.append(output)

        # Create the feature extraction model
        encoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder")
        encoder.trainable = True
        return encoder
    
    def _decoder(self, encoder_outputs, filters_list):     
        x = encoder_outputs[-1]
        for filters, skip, apply_dropout in zip(filters_list, encoder_outputs[-2::-1], [False] * 4):
            x = self._upsample_block(filters, 3)(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            x = self._conv_blocks(filters, size=3, apply_batch_norm=True, apply_dropout=apply_dropout)(x)
            x = self._conv_blocks(filters, size=1, apply_batch_norm=True)(x)
        return x
    
    def _conv_blocks(self, filters, size, apply_batch_norm=False, apply_instance_norm=False, apply_dropout=False):
        """Downsamples an input. Conv2D => Batchnorm => Dropout => LeakyReLU
            :param:
                filters: number of filters
                size: filter size
                apply_dropout: If True, adds the dropout layer
            :return: Downsample Sequential Model
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2D(filters, size, strides=1,
                                 padding='same', use_bias=False,
                                 kernel_initializer=initializer,))
        if apply_batch_norm:
            result.add(tf.keras.layers.BatchNormalization())
        if apply_instance_norm:
            result.add(tfa.layers.InstanceNormalization())
        result.add(tf.keras.layers.Activation(tfa.activations.mish))
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.55))
        return result
    
    def _upsample_block(self, filters, size, apply_dropout=False):
        """Upsamples an input. Conv2DTranspose => Batchnorm => Dropout => LeakyReLU
            :param:
                filters: number of filters
                size: filter size
                apply_dropout: If True, adds the dropout layer
            :return: Upsample Sequential Model
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.1))
        result.add(tf.keras.layers.Activation(tfa.activations.mish))
        return result


class IoU(tf.keras.metrics.Metric):
    def __init__(self, name='iou', **kwargs):
        super(IoU, self).__init__(name=name, **kwargs)
        self.confusion_matrix = self.add_weight(name='confusion_matrix', shape=(2, 2), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.math.round(tf.reshape(y_pred, [-1])), tf.float32)

        # Flatten the predictions and true labels to 1D arrays
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Compute confusion matrix
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2, dtype=tf.float32)

        # Update state
        self.confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        true_positives = self.confusion_matrix[1, 1]
        false_negatives = self.confusion_matrix[1, 0]
        false_positives = self.confusion_matrix[0, 1]

        intersection = true_positives
        union = true_positives + false_negatives + false_positives

        iou = intersection / (union + tf.keras.backend.epsilon())
        return iou

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self):
        config = {
            "confusion_matrix": self.confusion_matrix
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DiceScore(tf.keras.metrics.Metric):
    def __init__(self, name='dice_score', **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.confusion_matrix = self.add_weight(name='confusion_matrix', shape=(2, 2), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.math.round(tf.reshape(y_pred, [-1])), tf.float32)

        # Flatten the predictions and true labels to 1D arrays
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Compute confusion matrix
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2, dtype=tf.float32)

        # Update state
        self.confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        true_positives = self.confusion_matrix[1, 1]
        false_negatives = self.confusion_matrix[1, 0]
        false_positives = self.confusion_matrix[0, 1]

        dice_score = 2.0 * true_positives / (2.0 * true_positives + false_negatives + false_positives + tf.keras.backend.epsilon())
        return dice_score

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self):
        config = {
            "confusion_matrix": self.confusion_matrix
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])


df = pd.read_csv("airbus-ship-detection/train_ship_segmentations_v2.csv")
df['EncodedPixels'] = df['EncodedPixels'].astype('string')

# Delete corrupted images
CORRUPTED_IMAGES = ['6384c3e78.jpg']
df = df.drop(df[df['ImageId'].isin(CORRUPTED_IMAGES)].index)

# Dataframe that contains the segmentation for each ship in the image. 
instance_segmentation = df

# Dataframe that contains the segmentation of all ships in the image.
image_segmentation = df.groupby(by=['ImageId'])['EncodedPixels'].apply(lambda x: np.nan if pd.isna(x).any() else ' '.join(x)).reset_index()

IMAGES_WITHOUT_SHIPS_NUMBER = 10000

# reduce the number of images without ships
images_without_ships = image_segmentation[image_segmentation['EncodedPixels'].isna()]['ImageId'].values[:IMAGES_WITHOUT_SHIPS_NUMBER]
images_with_ships = image_segmentation[image_segmentation['EncodedPixels'].notna()]['ImageId'].values
images_list = np.append(images_without_ships, images_with_ships)

# remove corrupted images
images = list(filter(lambda x: x not in CORRUPTED_IMAGES, images_list))

VALIDATION_LENGTH = 20000
TRAIN_LENGTH = len(images) - VALIDATION_LENGTH
BATCH_SIZE = 16
BUFFER_SIZE = 100


def train_generator(images, dir=TRAIN_DIR):
    for img in images:
        # Load image
        img_path = os.path.join(dir, img)
        img_path = img_path.decode('utf-8')
        image = cv2.imread(img_path)
        input_image = cv2.resize(image, IMG_SHAPE)
        input_image = tf.cast(input_image, tf.float32) / 255.0

        encoded_mask = image_segmentation[image_segmentation['ImageId'] == img.decode('utf-8')].iloc[0]['EncodedPixels']
        input_mask = np.zeros(IMG_SHAPE + (1,), dtype=np.int8)
        if not pd.isna(encoded_mask):
            input_mask = rle_to_mask(encoded_mask)  
            input_mask = cv2.resize(input_mask, IMG_SHAPE, interpolation=cv2.INTER_AREA)
            input_mask = np.expand_dims(input_mask, axis=2)
        one_hot_segmentation_mask = one_hot(input_mask, NUM_CLASSES)
        input_mask_tensor = tf.convert_to_tensor(one_hot_segmentation_mask, dtype=tf.float32)

        yield input_image, input_mask_tensor

dataset = tf.data.Dataset.from_generator(train_generator, args=[images, TRAIN_DIR], output_types=(tf.float32, tf.float32), output_shapes=((256, 256, 3), (256, 256, 2)))

validation_dataset = dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE).repeat()
train_dataset = dataset.skip(VALIDATION_LENGTH).batch(BATCH_SIZE).repeat()


if __name__ == '__main__':
    EPOCHS = 3
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    loss = tf.keras.losses.CategoricalCrossentropy()
    dice = DiceScore()
    iou = IoU()

    model = UNetModel(IMG_SHAPE + (3,)).model
    model.compile(optimizer='adam', 
                loss=loss, # bce_dice_loss,
                metrics=[dice, iou,],)
    print("Everything's fine")
    model_history = model.fit(train_dataset,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=validation_dataset
                          )

    model.save('saved_models/my_model')



    


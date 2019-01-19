import keras
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import decode_predictions
from keras_applications import imagenet_utils
from keras_applications import get_keras_submodule
from keras_preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.layers import *
from keras import callbacks
from keras_contrib.applications.resnet import ResNet18
from scipy import ndimage
from PIL import Image
import tensorflow as tf
from keras import backend as K
from time import time
from keras.utils import multi_gpu_model
import os


train_dir = r''
val_dir = r''
epochs = 50
img_size = (320, 180)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = tf.Session(config=config)
    K.set_session(session)

    train_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.0,
                                       height_shift_range=0.0, brightness_range=(0.85, 1.15),
                                       shear_range=0.0, zoom_range=0.1,
                                       channel_shift_range=0.3,
                                       fill_mode='nearest', horizontal_flip=True,
                                       vertical_flip=False)
    valid_datagen = ImageDataGenerator()

    train = train_datagen.flow_from_directory(train_dir, target_size=img_size,
                                              color_mode='rgb', batch_size=32, interpolation='bicubic')
    valid = valid_datagen.flow_from_directory(val_dir, target_size=img_size,
                                              color_mode='rgb', batch_size=32, interpolation='bicubic')

    model = ResNet18(input_shape=(320, 180, 3), classes=2)

    
    learning_rate_base = 0.01
    sgd = keras.optimizers.SGD(
        lr=learning_rate_base, decay=learning_rate_base/100, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model_num = 1
    checkpoint_path = r''.format(model_num)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                     monitor='val_acc', verbose=1, save_best_only=True)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10,
                                            verbose=1, mode='auto', min_delta=0.0001,
                                            cooldown=0, min_lr=0)

    tensorboard = callbacks.TensorBoard(
        log_dir="logs/model_{}_{}".format(model_num, datetime.utcnow().strftime("%d%m%Y_%H%M%S")))

    model.fit_generator(train, steps_per_epoch=train.n/train.batch_size, epochs=epochs,
                        validation_data=valid, validation_steps=valid.n/valid.batch_size,
                        callbacks=[ckpt, reduce_lr, tensorboard])

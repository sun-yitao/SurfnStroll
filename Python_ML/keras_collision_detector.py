import keras
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import decode_predictions
from keras_applications import imagenet_utils
from keras_applications import get_keras_submodule
from keras_preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.layers import *
from keras import callbacks
#from keras.applications.inception_resnet_v2 import preprocess_input
#from keras_contrib.applications.wide_resnet import WideResidualNetwork
from scipy import ndimage
from PIL import Image
import tensorflow as tf
from keras import backend as K
from time import time
from keras.utils import multi_gpu_model

#K.clear_session()
import os

"""
Changelog
"""

train_dir = r''
val_dir = r''
epochs = 150
img_size = (200, 100)

BASE_WEIGHT_URL = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.7/')

k_regularizer = regularizers.l2(0.0001)
a_regularizer = regularizers.l2(0.0001)

#SpatialDropout2D(0.25)


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='prelu',
              use_bias=False,
              name=None, zero_gamma=False):

    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=k_regularizer,
                      kernel_initializer='he_normal',
                      #activity_regularizer=a_regularizer,
                      name=name)(x)

    #x = Dropout(0.1)(x)

    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        if zero_gamma == True:
            x = layers.BatchNormalization(axis=bn_axis,
                                          gamma_initializer='zeros',
                                          scale=False,
                                          name=bn_name)(x)
        else:
            x = layers.BatchNormalization(axis=bn_axis,
                                          scale=False,
                                          name=bn_name)(x)

    if activation == 'prelu':
        x = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None,
                         alpha_constraint=None, shared_axes=None)(x)
    elif activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)

    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='prelu'):

    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   backend.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   zero_gamma=True,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation == 'prelu':
        x = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None,
                         alpha_constraint=None, shared_axes=None)(x)
    elif activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      **kwargs):

    global backend, layers, models, keras_utils
    backend = get_keras_submodule('backend')
    engine = get_keras_submodule('engine')
    layers = get_keras_submodule('layers')
    models = get_keras_submodule('models')
    keras_utils = get_keras_submodule('utils')

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=75,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')
    #x = DropBlock2D(block_size=7, keep_prob=0.9)(x)
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = layers.MaxPooling2D(3, strides=2)(x)
    #x = SpatialDropout2D(0.25)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)
    #x = SpatialDropout2D(0.25)(x)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)
    #x = SpatialDropout2D(0.25)(x)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)

    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')
    #x = SpatialDropout2D(0.25)(x)

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='inception_resnet_v2')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            fname = ('inception_resnet_v2_weights_'
                     'tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.

    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=1):
        """Constructor for cosine decay with warmup learning rate scheduler.

    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages per epoch, 2: update messages per batch. (default: {1})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose == 1:
            lr = K.get_value(self.model.optimizer.lr)
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 1:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))

    class ModelMGPU(Model):
        def __init__(self, ser_model, gpus):
            pmodel = multi_gpu_model(ser_model, gpus)
            self.__dict__.update(pmodel.__dict__)
            self._smodel = ser_model

        def __getattribute__(self, attrname):
            '''Override load and save methods to be used from the serial-model. The
            serial-model holds references to the weights in the multi-gpu model.
            '''
            # return Model.__getattribute__(self, attrname)
            if 'load' in attrname or 'save' in attrname:
                return getattr(self._smodel, attrname)

            return super(ModelMGPU, self).__getattribute__(attrname)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = tf.Session(config=config)
    K.set_session(session)

    def preprocess(x):
        x -= 127.5
        x /= 127.5
        return x

    train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                       height_shift_range=0.1, brightness_range=(0.85, 1.15),
                                       shear_range=0.3, zoom_range=0.1,
                                       channel_shift_range=0.3,
                                       fill_mode='nearest', horizontal_flip=True,
                                       vertical_flip=False, preprocessing_function=preprocess)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train = train_datagen.flow_from_directory(train_dir, target_size=img_size,
                                              color_mode='rgb', batch_size=32, interpolation='bicubic')
    valid = valid_datagen.flow_from_directory(val_dir, target_size=img_size,
                                              color_mode='rgb', batch_size=32, interpolation='bicubic')

    input_tensor = keras.layers.Input(shape=(200, 100, 3))
    base_model = InceptionResNetV2(include_top=False,
                                   weights=None,
                                   input_tensor=input_tensor,
                                   input_shape=(200, 100, 3),
                                   pooling='avg',
                                   classes=2)
    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.25)(x)
    #x = Dense(1024, activation='relu', kernel_initializer='he_uniform')(x)
    predictions = Dense(2, activation='softmax',
                        kernel_regularizer=k_regularizer)(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    """
    #Multi GPU, call parallel model.fit_generator, may have to set save_weights_only=True and save architecture as json
    ser_model = Keras.models.Model(inputs=base_model.input, outputs=predictions)
    parallel_model = ModelMGPU(ser_model, 2)
    """
    learning_rate_base = 0.05
    sgd = keras.optimizers.SGD(
        lr=learning_rate_base, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = r':\yitao\changi\gender\checkpoints\model_29_checkpoints'
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                     monitor='val_acc', verbose=1, save_best_only=True)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10,
                                            verbose=1, mode='auto', min_delta=0.0001,
                                            cooldown=0, min_lr=0)

    tensorboard = callbacks.TensorBoard(log_dir="logs/{}".format(time()))

    # Number of warmup epochs.
    warmup_epoch = 5

    total_steps = int(epochs * train.n / train.batch_size)

    # Compute the number of warmup batches.
    warmup_steps = int(warmup_epoch * train.n / train.batch_size)
    warmup_batches = warmup_epoch * train.n / train.batch_size

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0)

    model.fit_generator(train, steps_per_epoch=train.n/train.batch_size, epochs=epochs,
                        validation_data=valid, validation_steps=valid.n/valid.batch_size,
                        callbacks=[ckpt, warm_up_lr, tensorboard])

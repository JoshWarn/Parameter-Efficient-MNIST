import tensorflow as tf
from keras.models import Model
from tensorflow_addons.optimizers import AdamW
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
import numpy as np

def s_conv_block(kernels, l, padding='valid'):
    l = SeparableConv2D(kernels, (3, 3), depth_multiplier=1, padding=padding, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(l)
    l = Dropout(rate=0.1)(l)
    return l


def model_function():
    inputs = Input((28, 28, 1), dtype=np.float32)
    l = tf.keras.layers.Rescaling(1./255., offset=0.0)(inputs)
    l = Conv2D(7, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2))(l)
    l = MaxPooling2D((2, 2))(l)
    l = Dropout(rate=0.1)(l)
    l = s_conv_block(10, l)
    l = s_conv_block(10, l)
    l = s_conv_block(11, l)
    l = s_conv_block(12, l)
    l = GlobalAveragePooling2D()(l)
    l = Dense(10, activation='softmax')(l)
    model = Model(inputs, l)
    op = AdamW(weight_decay=1e-6, learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=op, metrics=['accuracy'])

    trainableParams = int(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]))
    nonTrainableParams = int(np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights]))
    totalParams = int(trainableParams + nonTrainableParams)
    # print(f"Trainable: {trainableParams}, Non-Trainable: {nonTrainableParams}, Total: {totalParams}")
    return model, totalParams


(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test, axis=-1)

model, params = model_function()
print(params, print(len(x_test)))
model.load_weights("wei_997_0.9906.h5")
model.evaluate(x_test, y_test)

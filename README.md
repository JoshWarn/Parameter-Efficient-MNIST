# Parameter-Efficient-MNIST
How few parameters do you need to get 99% on MNIST? 5 convolution layers and 997 parameters later, we've found a maximum.
Inspired by https://github.com/ruslangrimov/mnist-minimal-model

![image](https://user-images.githubusercontent.com/70070682/230638687-9964ddbe-9684-4e3c-8004-5fd2acda878f.png)

```import tensorflow as tf
from keras.models import Model
from tensorflow_addons.optimizers import AdamW
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
from time import time


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


reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=8, verbose=0, mode="auto", min_delta=1.e-16, cooldown=0, min_lr=1e-8,)
early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=25, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
datagen = ImageDataGenerator()  # Used to shuffle training data
batch_size, num_epochs = 128, 200

model, params = model_function()
model.summary()
tf.keras.utils.plot_model(model, f"{params}__img.png", show_shapes=True)

while True:
    model, params = model_function()
    hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                        steps_per_epoch=x_train.shape[0] // batch_size, epochs=num_epochs,
                        validation_data=(x_test, y_test), verbose=0, callbacks=[reduce_lr, early_stop])

    saved_acc, max_acc = hist.history['val_accuracy'][-1], max(hist.history['val_accuracy'])
    print(f"Stop @ {len(hist.history['val_accuracy'])}; Best @ {hist.history['val_accuracy'].index(max_acc)}; "
          f"Saved Acc: {round(saved_acc, 6)}; Best Acc: {round(max_acc, 6)}")

    model.save(f"C:/Users/jwarn/tf_install/mini/{params}_{round(saved_acc, 7)}_{round(max_acc, 7)}_{int(time())}.h5")
    model.save_weights(f"C:/Users/jwarn/tf_install/wei_{params}_{round(saved_acc, 7)}.h5")

    with open(f"{params}_saved_acc.txt", "a+") as txtfile:
        txtfile.write(f"{saved_acc}, ")

    with open(f"{params}_max_acc.txt", "a+") as txtfile:
        txtfile.write(f"{max_acc}, ")

    del model
    tf.compat.v1.reset_default_graph
    tf.keras.backend.clear_session()
```

![image](https://user-images.githubusercontent.com/70070682/230638017-41fcfb78-babf-436d-adcf-5d9da45b472f.png)

import tensorflow as tf
from keras.models import Model
from tensorflow_addons.optimizers import AdamW
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
import numpy as np
import time
import os


class LrRecorder(tf.keras.callbacks.Callback):
    # records the lr after each epoch in the logs.
    def __init__(self, verbose=0):
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs):
        lr = self.model.optimizer.learning_rate.numpy()
        logs['lr'] = lr
        if self.verbose == 1:
            print(f"LR:{lr:.2E}")


class PercentileEarlyStopping(tf.keras.callbacks.Callback):
    # Stops networks after m epochs if they're below n-percentile of performers
    # E.g., after 50 epochs, if the model is in the bottom 20% of performers, stop it.
    # Params
    # # of epochs to train full-models before starting percentile-stopping
    # n epochs to wait before testing models
    # x-epoch smoothing of accuracy
    # p bottom p% that are removed/early-stopped
    def __init__(self, sample_size, epochs_before_testing, directory, prefix, suffix, percent_removed=0.5, epoch_smoothing=3, verbose=0):
        self.min_sample_size = sample_size
        self.epochs_before_testing = epochs_before_testing
        self.percent_to_remove = percent_removed
        self.epoch_smoothing = epoch_smoothing
        self.model_acc_list = []
        self.verbose = verbose
        self.cutoff_accuracy = None
        self.model_count = 0
        self.prefix = prefix
        self.suffix = suffix
        self.directory = directory

    def get_file_list(self):
        # TODO fix error of this also getting the "durations" and "acc" files...
        # ^^ should be fixed now, but haven't checked.
        data_file_list = []
        if self.directory != "":
            files_and_folders = os.listdir(self.directory)
            for item in files_and_folders:
                if item.startswith(self.prefix) and item.endswith(self.suffix):
                    if "acc" not in item and "durations" not in item:
                        data_file_list.append(f"{self.directory}/{item}")
        else:
            raise Exception(f"No directory provided! Directory: {self.directory}")
        return data_file_list

    def calculate_cutoff(self, datafilepaths):
        # TODO PROBABLY HUUUUGGGEEE ISSUE (possibly?)
        # TODO When a model is preemptive stopped, the log will remain.
        # TODO That means that they may affect future sample determinations if the script is restarted?
        model_accuracy_at_epoch = []
        for path in datafilepaths:
            f = open(path, "r")
            lines = f.readlines()
            header_list = lines[0].replace("\n", "").split(",")
            if "val_accuracy" in header_list:
                metric_index = header_list.index("val_accuracy")
            else:
                raise BaseException(f"ERR! Header does not have metric:{path} {header_list}")

            val_acc_list = []
            for i in range(len(lines[1:])):
                val_acc = lines[i + 1].replace("\n", "").split(",")[metric_index]
                val_acc_list.append(float(val_acc))
            # averaging the last n-epoch-smoothing models:

            # TODO Another another issue! This assumes that all previous models have been trained for at least self.epochs_before testing
            # which may not always be the case.
            # Right now, I'm simply going to exclude it if it's shorter than that:
            # I don't know if this works well...
            if len(val_acc_list) >= self.epochs_before_testing:
                sample_data = val_acc_list[self.epochs_before_testing - self.epoch_smoothing - 1: self.epochs_before_testing - 1]
                model_accuracy_at_epoch.append(sum(sample_data)/len(sample_data))

        # sort the list from min to max
        model_accuracy_at_epoch.sort()

        # calculate what index to grab:
        cutoff_index = int(self.percent_to_remove*len(model_accuracy_at_epoch))

        # get the percentile value (rounded down)
        cutoff_accuracy = model_accuracy_at_epoch[cutoff_index]
        return cutoff_accuracy

    def on_epoch_end(self, epoch, logs):
        # Maintains a list of the last n-epochs val-accuracies for n-epoch smoothing..
        val_acc = logs["val_accuracy"]
        self.model_acc_list.append(val_acc)
        # Epoch is testing epoch:
        if epoch == self.epochs_before_testing:
            data_file_list = self.get_file_list()   # list of datafiles
            if len(data_file_list) > self.min_sample_size:

                # Calculate the cutoff accuracy from the sample-set
                self.cutoff_accuracy = self.calculate_cutoff(data_file_list)

                # Calculating last-n-epoch-average accuracy
                model_val_acc = sum(self.model_acc_list[-self.epoch_smoothing:]) / len(self.model_acc_list[-self.epoch_smoothing:])

                if model_val_acc <= self.cutoff_accuracy:
                    if self.verbose == 1:
                        print(f"EarlyPercentileStop: {epoch}: Acc {model_val_acc} below {self.cutoff_accuracy}.")
                    self.model.stop_training = True
                else:
                    if self.verbose == 1:
                        print(f"EarlyPercentileStop: {epoch}: Acc {model_val_acc} above {self.cutoff_accuracy}. Cont.")



def training_and_evaluating(model, batch_size, num_epochs, parameter_count, callback_list, datasetname, dataset):
    model.summary()
    tf.keras.utils.plot_model(model, f"{parameter_count}_img.png", show_shapes=True)

    (x_train, y_train), (x_test, y_test) = dataset
    x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
    datagen = ImageDataGenerator()  # Used to shuffle training data

    # Make a folder to put all the stuff in
    folder_path = f"{parameter_count}_{datasetname}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    i = 1
    while True:
        start = time.time()
        csv_log = CSVLogger(f"{folder_path}{parameter_count}_{datasetname}_{i:03d}_{int(start)}.txt")

        model, params = model_function()
        hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                         steps_per_epoch=x_train.shape[0] // batch_size, epochs=num_epochs,
                         validation_data=(x_test, y_test), verbose=1, callbacks=callback_list + [csv_log])

        acc_list = hist.history['val_accuracy']
        max_acc = max(acc_list)
        print(f"Stop @ {len(acc_list)}; Best @ {acc_list.index(max_acc)}: {round(max_acc, 6)}")

        model.save_weights(f"{folder_path}{parameter_count}_{datasetname}_{i:03d}_{int(start)}_{round(max_acc, 7)}.h5")

        with open(f"{folder_path}{parameter_count}_{datasetname}.acc", "a+") as txtfile:
            txtfile.write(f"{max_acc}, ")

        with open(f"{folder_path}{parameter_count}_{datasetname}.dur", "a+") as txtfile:
            txtfile.write(f"{time.time() - start}, ")

        del model
        tf.compat.v1.reset_default_graph
        tf.keras.backend.clear_session()
        i += 1


def s_conv_block(kernels, l, padding='valid', dropout=True):
    l = SeparableConv2D(kernels, (3, 3), depth_multiplier=1, padding=padding,
                        activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                        kernel_initializer=tf.keras.initializers.he_normal)(l)
    if dropout:
        l = Dropout(rate=0.05)(l)
    return l


def model_function():
    inputs = Input((28, 28, 1), dtype=np.float32)
    l = tf.keras.layers.Rescaling(1./255., offset=0.0)(inputs)
    l = Conv2D(6, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.he_normal, padding='same')(l)
    l = MaxPooling2D((2, 2))(l)
    l = s_conv_block(5, l)
    l = s_conv_block(9, l)                  # 9 works; 8 likely not.
    l = s_conv_block(10, l)                 # Has to be 10 it seems.
    l = s_conv_block(8, l, dropout=False)   # 8 may be possible. 9 works.
    l = GlobalAveragePooling2D()(l)
    l = Dense(10, activation='softmax')(l)
    model = Model(inputs, l)
    op = AdamW(weight_decay=1e-6, learning_rate=1e-2)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=op, metrics=['accuracy'])

    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    print(f"Trainable: {trainableParams}, Non-Trainable: {nonTrainableParams}, Total: {totalParams}")
    return model, totalParams


reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=50, verbose=1, mode="auto", min_delta=1.e-16, cooldown=0, min_lr=2.5e-5,)   # Normally set to 50.
early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=150, verbose=1, mode='auto', baseline=None, restore_best_weights=True)     # Normally set to 150.
percentile_early_stop = PercentileEarlyStopping(30, 100, "697.0_mnist", "697.0_mnist_", ".txt", percent_removed=0.5, epoch_smoothing=3, verbose=1)
record_lr = LrRecorder(verbose=0)

# With a LR of 5e-3 after 20 epochs-3-avg we can drop the bottom 20%?
# TODO Calculate what the bottom-20-percent cutoff is...
# OOOORRRR after 40 epochs-3-avg cut the bottom 50%. would save a ton of time long-term.

# Determine what the bottom-50% is by taking a 30-test sample(?)

callbacks = [reduce_lr, early_stop, percentile_early_stop, record_lr]  # CSVLogger is added in the training_and_evaluating function.
batch_size, num_epochs = 256, 2000
model, params = model_function()
datasetname, dataset = "mnist", tf.keras.datasets.mnist.load_data()
training_and_evaluating(model, batch_size, num_epochs, params, callbacks, datasetname, dataset)
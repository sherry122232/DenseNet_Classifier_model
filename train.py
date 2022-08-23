# --coding:utf-8--
import os
import sys
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.densenet import DenseNet201, preprocess_input
from keras.models import Model
from keras.layers import
, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


# Dataset prepare
IM_WIDTH, IM_HEIGHT = 224, 224

train_dir = './data/train'  # training dataset dir
val_dir = './data/test'  # validate dataset dir
nb_classes = 10
nb_epoch = 10
batch_size = 8

nb_train_samples = get_nb_files(train_dir)  # number of train_samples
nb_classes = len(glob.glob(train_dir + "/*"))  # number of classes
nb_val_samples = get_nb_files(val_dir)  # number of validate samples
nb_epoch = int(nb_epoch)  # number of epoch
batch_size = int(batch_size)

# 　generate pictures
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True
)

# training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical')


# add new layer
def add_new_last_layer(base_model, nb_classes):
    """
    input:
    base_model && number of classes
    output:
    new model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='sigmoid')(x)  # new softmax layer

    model = Model(input=base_model.input, output=predictions)
    return model


# Build the model
model = DenseNet201(include_top=False)
model = add_new_last_layer(model, nb_classes)

model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True), loss='categorical_crossentropy',
              metrics=['accuracy'])


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'model/checkpoint-01e-val_accuracy_0.95.hdf5', monitor='val_loss', verbose=0,
    save_best_only=False, save_weights_only=False, mode='auto', period=1)

# 第二次训练可以接着第一次训练得到的模型接着训练
# model.load_weights('D:/week/model/checkpoint-01e-val_accuracy_0.95.hdf5')

# 更好地保存模型 Save the model after every epoch.
output_model_file = 'checkpoint-new{epoch:02d}e-val_accuracy_{' \
                    'val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(output_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)
#
# starting training
history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    callbacks=[checkpoint],
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples)


def plot_training(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'r-')
    plt.plot(epochs, val_accuracy, 'b')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'r-')
    plt.plot(epochs, val_loss, 'b-')
    plt.title('Training and validation loss')
    plt.show()


# 训练的acc_loss图
plot_training(history_ft)

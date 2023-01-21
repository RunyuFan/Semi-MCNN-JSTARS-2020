from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model, Input
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Multiply,GlobalAveragePooling2D, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Dropout, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(56, 56, 3))  # this assumes K.image_data_format() == 'channels_last'

num_classes = 4
img_rows, img_cols = 56, 56
img_width, img_height = 56, 56
train_data_dir ='.\\Shenzhen56-5-5-gen\\train_data'
validation_data_dir ='.\\Shenzhen56-5-5-gen\\val_data'
Imagesize = 56
epochs = 100
batch_size = 32
channel = 3


# create the base pre-trained model
base_model = ResNet50(input_tensor=input_tensor, weights=None, include_top=False)
# base_model.summary()
# base_model.load_weights('./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
x_newfc = base_model.output

x_newfc =  GlobalAveragePooling2D()(x_newfc)
# x_newfc = Convolution2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(x_newfc)
# x_newfc = Convolution2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(x_newfc)
# x_newfc = Flatten()(x_newfc)
# x_newfc = Dropout(0.5)(x_newfc)
x = Dense(num_classes, activation="softmax")(x_newfc)
img_input = Input(shape=(img_rows, img_cols, channel))
model = Model(base_model.input, x)
sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)
    #class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)
    #class_mode='binary')
image_numbers = train_generator.samples
print(train_generator.class_indices)

history_object = model.fit_generator(train_generator,steps_per_epoch = image_numbers // batch_size, epochs = epochs, validation_data = validation_generator, validation_steps = batch_size)
'''
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=batch_size)
'''
model.save_weights('Res50-student-56-5-5.h5')
score = model.evaluate_generator(validation_generator, len(validation_generator), verbose=1)
print(score)
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
# plt.plot(history_object.history['acc'], label="train fine tune")
plt.plot(history_object.history['val_acc'], color='green', label="Accuracy")
# plt.plot(history_object2.history['val_acc'], color='blue', label="Accuracy of AlexNet")
plt.ylabel('Test Accuracy')
plt.xlabel('Number of Training Epochs')
plt.ylim((0, 1.2))
plt.grid(True)
plt.legend(loc='upper left')
# plt.savefig('./acc-BNAlex-NWPUfinetune.jpg')
# plt.show()

plt.twinx()  # 添加一条Y轴，
# plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'], color='red', ls='-.', label="Loss")
# plt.plot(history_object2.history['val_loss'], color='yellow', ls='-.', label="Loss of AlexNet")
plt.ylabel('Test Loss')
plt.legend(loc='upper right')  #  loc='best'
# plt.legend(bbox_to_anchor=(0., 1.15),ncol=2)
plt.savefig('./NWPU50.jpg')
plt.show()
'''

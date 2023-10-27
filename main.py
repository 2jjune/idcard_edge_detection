import random

import matplotlib.pyplot as plt
import cv2
import requests
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
# from tensorflow.keras.prerprocessing.image import ImageDataGenerator
# from tensorflow import keras
import edge_detection_model
from focal_loss import BinaryFocalLoss

batch_size = 4
img_height = 256
img_width = 256
# dir = 'C://Users//USER//PycharmProjects//img_dataset_for_edge_detection//imgs//train'

base_dir = 'C:/Users/dlwld/PycharmProjects/edge_detection/image_dataset_for_edge_detection/'
train_dir = os.path.join(base_dir + 'imgs/train/')
# train_label_dir = os.path.join(base_dir + 'edge_maps/train/')
train_label_dir = os.path.join(base_dir + 'edge_maps_filled/train/')
val_dir = os.path.join(base_dir + 'imgs/val/')
# val_label_dir = os.path.join(base_dir + 'edge_maps/val/')
val_label_dir = os.path.join(base_dir + 'edge_maps_filled/val/')

train_img_list = os.listdir(train_dir)
train_label_list = os.listdir(train_label_dir)
val_img_list = os.listdir(val_dir)
val_label_list = os.listdir(val_label_dir)

#-------------------------------------------------------------------------------------------------
def train_gen():
    # for i in range(0, len(train_img_list) - batch_size + 1, batch_size):

    train_img_list = os.listdir(train_dir)
    while len(train_img_list)>batch_size:
        img_list = []
        label_list = []
        for j in range(batch_size):

            pop_img_file = train_img_list.pop(random.randrange(0, len(train_img_list)))

            ####augmentation 해보기


            ########################
            img_file = cv2.imread(train_dir + pop_img_file)
            img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2HSV)
            label_file = cv2.imread(train_label_dir + pop_img_file, 0)
            # img_file = cv2.imread(train_dir + train_img_list[i + j])

            img_file = cv2.resize(img_file, (img_height, img_width))
            label_file = cv2.resize(label_file, (img_height, img_width), interpolation=cv2.INTER_NEAREST)

            # img_file = np.asarray(img_file)
            # label_file = np.asarray(label_file)

            img_file = img_file / 255.
            label_file = label_file / 255.
            label_file = np.expand_dims(label_file, axis=2)



            img_list.append(img_file)
            label_list.append(label_file)
            # print('/'*100)
            # print(np.asarray(label_list).shape)
            # print(np.expand_dims(np.asarray(label_list),axis=3).shape)
            # print('/'*100)
            # label_list = np.expand_dims(label_list,axis=3)
            # print('/'*100)
            #
            # label_list.reshape(img_height,img_width,1)
            # print(np.asarray(label_list).shape)
            # print('/'*100)


        yield (img_list, label_list)


def val_gen():
    for i in range(0, len(val_img_list) - batch_size + 1, batch_size):

        img_list = []
        label_list = []
        for j in range(batch_size):
            img_file = cv2.imread(val_dir + val_img_list[i + j])
            img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2HSV)
            img_file = cv2.resize(img_file, (img_height, img_width))
            label_file = cv2.imread(val_label_dir + val_label_list[i + j],0)
            label_file = cv2.resize(label_file, (img_height, img_width), interpolation=cv2.INTER_NEAREST)

            # img_file = np.asarray(img_file)
            # label_file = np.asarray(label_file)

            img_file = img_file / 255.
            label_file = label_file / 255.
            label_file = np.expand_dims(label_file, axis=2)

            img_list.append(img_file)
            label_list.append(label_file)
            # label_list.reshape(img_height,img_width,1)
            # label_list = np.expand_dims(label_list,axis=3)
        yield (img_list, label_list)

#-----------------------------------------------------------------------------------------------




train_data = tf.data.Dataset.from_generator(train_gen, (tf.float32, tf.float32), ((batch_size,img_height,img_width,3), (batch_size,img_height,img_width,1)))
val_data = tf.data.Dataset.from_generator(val_gen, (tf.float32, tf.float32), ((batch_size,img_height,img_width,3), (batch_size,img_height,img_width,1)))



callbacks = [
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
             tf.keras.callbacks.ModelCheckpoint('./checkpoint/best.h5', monitor = 'val_loss', save_weights_only=True,save_best_only=True)
]

# model = edge_detection_model.UNet()
# model = edge_detection_model.UNetCompiled_skip_small(input_size=(img_height,img_width,3), n_filters=64, n_classes=1)
# base_model = tf.keras.applications.vgg16.VGG16(
#     include_top=False, input_shape=(img_width, img_height, 3))
# base_model_resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(img_width, img_height, 3))

# layer_names = [
#     'block1_pool',
#     'block2_pool',
#     'block3_pool',
#     'block4_pool',
#     'block5_pool',
# ]
#
# resnet_layer_names = ['input_1', 'conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
# base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
# base_model.trainable = False
#
#
# VGG_16 = tf.keras.models.Model(base_model.input, base_model_outputs)
#
# model = edge_detection_model.segmentation_model(img_width,img_height,VGG_16)
model = edge_detection_model.build_resnet50_unet((img_width,img_height,3))
model.summary()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['accuracy'])
history = model.fit(train_data, epochs=500, shuffle=True, validation_data=val_data, callbacks=callbacks)

def history_chart(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()

history_chart(history)


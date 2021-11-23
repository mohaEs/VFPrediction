#!/usr/bin/env python
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, GaussianNoise, Flatten, Dense, Reshape, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose , LeakyReLU, Activation, Add, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import json, glob, random, sys
import losses

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
def getModel(name, shape):
    if name == "Identity":
        return get_ident(shape)
    if name == "FC":
        return get_fc(shape)
    if "-" in name:
        mod = name.split("-")[0]
        fil = int(name.split("-")[1][0])
    if mod == "DilCN":
        return get_dilcn(shape, fil)
    if mod == "FullCN":
        return get_fullcn(shape, fil)
    if mod == "FullBN":
        return get_fullbn(shape, fil)
    if mod == "Residual":
        return get_resnet(shape, fil)
    if mod == "CascadeNet":
        return get_cascadenet(shape, fil)
    print("Unknown model: %s ") #, % name 
    sys.exit()


def get_ident(shape):
    img_rows, img_cols, channels = shape[0], shape[1], shape[2]
    inputs = Input((img_rows, img_cols, channels))
    output = inputs

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=1e-3), loss=losses.mae)
    return model


def get_fc(shape):
    img_rows, img_cols, channels = shape[0], shape[1], shape[2]
    inputs = Input((img_rows, img_cols, channels))
    noise = GaussianNoise(0.5)(inputs)
    flat = Flatten()(noise)
    dense = Dense(512, activation='relu')(flat)
    dense = Dense(512, activation='relu')(dense)
    dense = Dense(img_rows * img_cols)(dense)
    output = Reshape((img_rows, img_cols, 1))(dense)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=1e-3), loss=losses.mae)
    return model


def get_dilcn(shape, num_cn):
    img_rows, img_cols, channels = shape[0], shape[1], shape[2]
    inputs = Input((img_rows, img_cols, channels))
    conv1 = GaussianNoise(0.5)(inputs)
    #conv1 = inputs
    for i in range(num_cn):
      conv1 = Conv2D(64, (3, 3), activation='relu',  padding='same')(conv1)
      #conv1 = LeakyReLU()(conv1)
      #conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(64, (3, 3), dilation_rate=2, padding='same')(conv1)
    for i in range(num_cn):
      conv2 = Conv2D(128, (3, 3),  activation='relu',padding='same')(conv2)
      #conv2 = LeakyReLU()(conv2)
      #conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(128, (3, 3), dilation_rate=2, padding='same')(conv2)
    for i in range(num_cn):
      conv3 = Conv2D(256, (3, 3),  activation='relu',padding='same')(conv3)
      #conv3 = LeakyReLU()(conv3)
      #conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), dilation_rate=2, padding='same')(conv3)

    conv4 = Conv2D(1, (1, 1))( conv4)

    model = Model(inputs=[inputs], outputs=[conv4])
    model.compile(optimizer=Adam(lr=1e-5), loss=losses.mae)
    return model

def cascade_block(x, filters, size):
      x = BatchNormalization(axis=-1)(x)
      x = Activation("relu")(x)
      x = Conv2D(filters, (size, size), padding='same', kernel_initializer="he_normal")(x)
      return x

def get_cascadenet(shape, num_cn):
    img_rows, img_cols, channels = shape[0], shape[1], shape[2]
    input = Input((img_rows, img_cols, channels))
    shortcuts = [input]
    x = input
    for i in range(num_cn):
      x = cascade_block(x, 64, 3)
      shortcuts.append(x)
      x = Concatenate()(shortcuts)
    for i in range(num_cn):
      x = cascade_block(x, 128, 3)
      shortcuts.append(x)
      x = Concatenate()(shortcuts)
    for i in range(num_cn):
      x = cascade_block(x, 256, 3)
      shortcuts.append(x)
      x = Concatenate()(shortcuts)
    conv4 = cascade_block(x, 1, 1)
    model = Model(inputs=[input], outputs=[conv4])
    model.compile(optimizer=Adam(lr=1e-3), loss=losses.mae)
    return model
    


def res_block(x, filters, size, last=False):
      shortcut = x
      x = BatchNormalization(axis=-1)(x)
      x = Activation("relu")(x)
      x = Conv2D(filters, (size, size), padding='same', kernel_initializer="he_normal")(x)
      if last:
        return x
      return Add()([shortcut, x])


def get_resnet(shape, num_cn):
    img_rows, img_cols, channels = shape[0], shape[1], shape[2]
    inputs = Input((img_rows, img_cols, channels))
    conv1 = inputs
    #conv1 = GaussianNoise(0.5)(inputs)
    #conv1 = inputs
    for i in range(num_cn):
      conv1 = res_block(conv1, 64, 3)

    conv2 = Conv2D(128, (1, 1), padding="same", kernel_initializer="he_normal")(conv1)
    for i in range(num_cn):
      conv2 = res_block(conv2, 128, 3)

    conv3 = Conv2D(256, (1, 1), padding="same", kernel_initializer="he_normal")(conv2)
    for i in range(num_cn):
      conv3 = res_block(conv3, 256, 3)

    conv4 = res_block(conv3, 1, 1, True)

    model = Model(inputs=[inputs], outputs=[conv4])
    model.compile(optimizer=Adam(lr=1e-3), loss=losses.mae)
    return model


def get_fullbn(shape, num_cn):
    img_rows, img_cols, channels = shape[0], shape[1], shape[2]
    inputs = Input((img_rows, img_cols, channels))
    conv1 = inputs
    conv1 = BatchNormalization(axis=-1)(conv1)
    #conv1 = GaussianNoise(0.5)(inputs)
    #conv1 = inputs
    for i in range(num_cn):
      conv1 = Conv2D(64, (3, 3), activation='relu',  padding='same')(conv1)
      #conv1 = LeakyReLU()(conv1)
      conv1 = BatchNormalization(axis=-1)(conv1)

    conv2 = conv1
    for i in range(num_cn):
      conv2 = Conv2D(128, (3, 3),  activation='relu',padding='same')(conv2)
      #conv2 = LeakyReLU()(conv2)
      conv2 = BatchNormalization(axis=-1)(conv2)

    conv3 = conv2
    for i in range(num_cn):
      conv3 = Conv2D(256, (3, 3),  activation='relu',padding='same')(conv3)
      #conv3 = LeakyReLU()(conv3)
      conv3 = BatchNormalization(axis=-1)(conv3)

    conv4 = Conv2D(1, (1, 1))( conv3)

    model = Model(inputs=[inputs], outputs=[conv4])
    model.compile(optimizer=Adam(lr=1e-3), loss=losses.mae)
    return model


def get_fullcn(shape, num_cn):
    img_rows, img_cols, channels = shape[0], shape[1], shape[2]
    inputs = Input((img_rows, img_cols, channels))
    conv1 = inputs
    #conv1 = GaussianNoise(0.5)(inputs)
    #conv1 = inputs
    for i in range(num_cn):
      conv1 = Conv2D(64, (3, 3), activation='relu',  padding='same')(conv1)
      #conv1 = LeakyReLU()(conv1)
      #conv1 = BatchNormalization()(conv1)

    conv2 = conv1
    for i in range(num_cn):
      conv2 = Conv2D(128, (3, 3),  activation='relu',padding='same')(conv2)
      #conv2 = LeakyReLU()(conv2)
      #conv2 = BatchNormalization()(conv2)

    conv3 = conv2
    for i in range(num_cn):
      conv3 = Conv2D(256, (3, 3),  activation='relu',padding='same')(conv3)
      #conv3 = LeakyReLU()(conv3)
      #conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(1, (1, 1))( conv3)

    model = Model(inputs=[inputs], outputs=[conv4])
    model.compile(optimizer=Adam(lr=1e-5), loss=losses.mae)
    return model




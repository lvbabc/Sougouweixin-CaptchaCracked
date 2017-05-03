# -*- coding: utf-8 -*-
import numpy as np
import random
import string
from keras.utils.np_utils import to_categorical
from keras.models import *
from keras.layers import *
from tqdm import tqdm
import h5py
import os
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt

characters = string.digits + string.ascii_uppercase

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    while True:
        for i in range(batch_size):
            num = random.randint(0, image_size-1)
            random_str = image_names[num]
            X[i] = image_array[num]
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


# 定义网络结构
input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32 * 2 ** i, (3, 3), activation='relu')(x)
    x = Convolution2D(32 * 2 ** i, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(6)]
model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit_generator(gen(), steps_per_epoch=51200, epochs=10, workers=2, pickle_safe=True, validation_data=gen(), validation_steps=1280)


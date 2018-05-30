from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D,Cropping2D
from keras.layers.normalization import BatchNormalization
import json

def build_model():
    img_w = im_width
    img_h = im_height
    n_labels = 2
    kernel = 3

    encoding_layers = [
        Lambda(lambda x: x / 255.0 - 0.5, input_shape = (img_h, img_w, 3)),
        Conv2D(12, (kernel,kernel), padding='same', activation = "relu"),#, input_shape=( img_h, img_w,3)),
        BatchNormalization(),
        Conv2D(12, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        ZeroPadding2D(((0,0),(0,1))),
        MaxPooling2D(),

        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        ZeroPadding2D(((1,0),(1,1))),
        MaxPooling2D(),

        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        ZeroPadding2D(((1,0),(0,0))),
        MaxPooling2D(),

        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        MaxPooling2D(),

    ]

    model = models.Sequential()
    model.encoding_layers = encoding_layers
    for l in model.encoding_layers:
        model.add(l)

    decoding_layers = [

        UpSampling2D(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),

        UpSampling2D(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(64, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Cropping2D(((1,0),(0,0))),

        UpSampling2D(),
        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Cropping2D(((1,0),(0,1))),

        UpSampling2D(),
        Conv2D(24, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(12, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Cropping2D(((0,0),(0,1))),

        UpSampling2D(),
        Conv2D(12, (kernel,kernel), padding='same', activation = "relu"),
        BatchNormalization(),
        Conv2D(n_labels, 1, padding='valid'),
        BatchNormalization(),
        Flatten(),
        Activation('sigmoid')

    ]
    model.decoding_layers = decoding_layers
    for l in model.decoding_layers:
        model.add(l)
    return model
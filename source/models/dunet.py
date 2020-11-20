import tensorflow as tf
from tensorflow.keras import models,layers

### A Unet-like architecture with a ASPP module at the end of the encoder ###

def dunet(height=560,width=400,nbr_mask=10,nbr=16, activation='softmax'):
    
    #Inputimage
    entree = layers.Input(shape=(height, width, 3), dtype='float32')

    #Level 0
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(entree)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result1 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result1)

    #Level -1
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result2 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result2)

    #Level -2
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result3 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result3)

    # Level -3
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    # ASPP
    b0 = layers.DepthwiseConv2D(3, activation='relu', padding='same')(result)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b0)
    b0 = layers.BatchNormalization()(b0)

    b1 = layers.DepthwiseConv2D(3, dilation_rate=(6, 6), activation='relu', padding='same')(result)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b1)
    b1 = layers.BatchNormalization()(b1)

    b2 = layers.DepthwiseConv2D(3, dilation_rate=(12, 12), activation='relu', padding='same')(result)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b2)
    b2 = layers.BatchNormalization()(b2)

    b3 = layers.DepthwiseConv2D(3, dilation_rate=(18, 18), activation='relu', padding='same')(result)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b3)
    b3 = layers.BatchNormalization()(b3)

    b4 = layers.AveragePooling2D()(result)
    b4 = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.UpSampling2D(interpolation='bilinear')(b4)

    result = layers.Concatenate()([b4, b0, b1, b2, b3])

    result = layers.Conv2D(8 * nbr, 1, activation='relu', padding='same')(result)

    result = layers.Conv2DTranspose(4 * nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)

    # Level -2
    result = tf.concat([result, result3], axis=3)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(2 * nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)

    # Level -1
    result = tf.concat([result, result2], axis=3)
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)

    # Level 0
    result = tf.concat([result, result1], axis=3)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    # Output
    sortie = layers.Conv2D(nbr_mask, 1, activation=activation, padding='same')(result)

    model = models.Model(inputs=entree, outputs=sortie)
    return model
    
print(dunet().summary())

import tensorflow as tf
from tensorflow.keras import layers, models

### A lighter attempt, less parameters than in customnetv1###
### Not performant with our data ###

def customnetv2(height=560, width=400, nbr_mask=10, nbr=16, activation='softmax'):
    # Inputimage
    entree = layers.Input(shape=(height, width, 3), dtype='float32')

    # Level 0
    result = layers.Conv2D(nbr, 7, activation='relu', padding='same')(entree)
    result1 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result1)

    # Level -1
    result = layers.Conv2D(2 * nbr, 5, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result)

    # Level -2
    result = layers.Conv2D(4* nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)


    # ASPP
    b0 = layers.DepthwiseConv2D(3, activation='relu', padding='same')(result)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Conv2D(4 * nbr, 1, activation='relu', padding='same')(b0)
    b0 = layers.BatchNormalization()(b0)

    b1 = layers.DepthwiseConv2D(3, dilation_rate=(6, 6), activation='relu', padding='same')(result)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Conv2D(nbr, 1, activation='relu', padding='same')(b1)
    b1 = layers.BatchNormalization()(b1)

    b2 = layers.DepthwiseConv2D(3, dilation_rate=(18, 18), activation='relu', padding='same')(result)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Conv2D(4 * nbr, 1, activation='relu', padding='same')(b2)
    b2 = layers.BatchNormalization()(b2)

    b3 = layers.DepthwiseConv2D(3, dilation_rate=(36, 36), activation='relu', padding='same')(result)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Conv2D(4 * nbr, 1, activation='relu', padding='same')(b3)
    b3 = layers.BatchNormalization()(b3)

    b4 = layers.AveragePooling2D()(result)
    b4 = layers.Conv2D(4 * nbr, 1, activation='relu', padding='same')(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.UpSampling2D(interpolation='bilinear')(b4)

    result = layers.Concatenate()([b4, b0, b1, b2, b3])

    result = layers.Conv2D(4 * nbr, 1, activation='relu', padding='same')(result)

    # ending
    result = layers.Conv2DTranspose(nbr, 4, strides=(4, 4), activation='relu', padding='same')(result)
    result = layers.Concatenate()([result1,result])
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result=layers.BatchNormalization()(result)

    sortie = layers.Conv2D(nbr_mask, 1, activation=activation, padding='same')(result)

    model = models.Model(inputs=entree, outputs=sortie)
    return model
    
print(customnetv2().summary())

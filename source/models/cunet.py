import tensorflow as tf
from tensorflow.keras import layers, models

### Use of Depthwise convolution to reduce the number of parameters in the encoder ###
### Remove of skip connections between Encoder and Decoder to reduce memory usage ###
### go less deeper than original Unet ###

def cunet(height=560,width=400,nbr_mask=10,nbr=16,activation='softmax'):
    #Inputimage
    entree = layers.Input(shape=(height, width, 3), dtype='float32')

    #Level 0
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(entree)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.SeparableConv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result)

    #Level -1
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.SeparableConv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result)

    #Level -2
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.SeparableConv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result)

    # Level -3
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.SeparableConv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(4 * nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)

    # Level -2
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(2 * nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)

    # Level -1
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(nbr, 2, strides=(2, 2), activation='relu', padding='same')(result)

    # Level 0
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    # Output
    sortie = layers.Conv2D(nbr_mask, 1, activation=activation, padding='same')(result)

    model = models.Model(inputs=entree, outputs=sortie)
    return model
 
 print(cunet().summary())

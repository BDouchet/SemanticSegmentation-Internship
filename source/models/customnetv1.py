import tensorflow as tf
from tensorflow.keras import layers,models

### An attempt to create a light model for semantic segmentation ###
### Not really performant with our datas ###

def customnetv1(height=560,width=400,nbr_mask=10,nbr=16,activation='softmax'):
    # Inputimage
    entree = layers.Input(shape=(height, width, 3), dtype='float32')

    # Level 0
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(entree)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result1 = layers.MaxPool2D()(result)

    # Level -1
    result = layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result1)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(2*nbr, 3, activation='relu', padding='same')(result)
    result2 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result2)

    # Level -2
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result3 = layers.BatchNormalization()(result)

    result = layers.MaxPool2D()(result3)

    # Level -3
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result4 = layers.BatchNormalization()(result)

    #upsample
    result3 = layers.Conv2DTranspose(4 * nbr, 2, strides=(2, 2), activation='relu', padding='same')(result3)

    result4 = layers.Conv2DTranspose(8 * nbr, 2, strides=(2, 2), activation='relu', padding='same')(result4)
    result4 = layers.Conv2DTranspose(8 * nbr, 2, strides=(2, 2), activation='relu', padding='same')(result4)

    #ending
    result = layers.Concatenate()([result1, result2, result3,result4])
    result = layers.Conv2D(2*nbr, 1, activation=activation, padding='same')(result)
    result = layers.BatchNormalization()(result)

    result = layers.Conv2DTranspose(2*nbr,2, strides=(2, 2), activation='relu', padding='same')(result)

    result = layers.Conv2D(nbr, 5, activation=activation, padding='same')(result)
    result = layers.BatchNormalization()(result)

    sortie = layers.Conv2D(nbr_mask, 1, activation=activation, padding='same')(result)

    model = models.Model(inputs=entree, outputs=sortie)
    return model

print(customnetv1().summary())

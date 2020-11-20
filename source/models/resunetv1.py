import tensorflow as tf
from tensorflow.keras import layers, models

### Use of more convolutionnal layers with residual blocks ###
### No skip connections ###

def resunetv1(height=560, width=400, nbr_mask=10, nbr=16, activation='softmax'):
    # Inputimage
    entree = layers.Input(shape=(height, width, 3), dtype='float32')

    #Level0
    result = layers.Conv2D(nbr,3,activation='relu',padding='same')(entree)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    result=layers.MaxPool2D()(result)

    #level-1
    result=layers.Conv2D(2*nbr,3,activation='relu',padding='same')(result)
    result=layers.BatchNormalization()(result)
    result = resblock(result,3,2*nbr)
    result = resblock(result, 3, 2 * nbr)
    result=layers.MaxPool2D()(result)

    #Level -2
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result=resblock(result,3,4*nbr)
    result = resblock(result, 3, 4 * nbr)
    result=layers.MaxPool2D()(result)

    #Level-3
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = resblock(result, 3, 8 * nbr)
    result = resblock(result, 3, 8 * nbr)
    result = layers.UpSampling2D()(result)

    #Level -2
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = resblock(result, 3, 4 * nbr)
    result = resblock(result, 3, 4 * nbr)
    result = layers.UpSampling2D()(result)

    #Level-1
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = resblock(result, 3, 2 * nbr)
    result = resblock(result, 3, 2 * nbr)
    result = layers.UpSampling2D()(result)

    #Level0
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)

    #Output
    sortie = layers.Conv2D(nbr_mask, 1, activation=activation, padding='same')(result)

    model = models.Model(inputs=entree, outputs=sortie)
    return model
    
print(resunetv1().summary())

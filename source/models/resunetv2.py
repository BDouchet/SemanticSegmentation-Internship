import tensorflow as tf
from tensorflow.keras import layers, models

### similar to resunetv1 with more convolutionnal layers and skip connections ###

def resunetv2(height=560, width=400, nbr_mask=10, nbr=16, activation='softmax'):
    # Inputimage
    entree = layers.Input(shape=(height, width, 3), dtype='float32')

    #Level0
    result = layers.Conv2D(nbr,3,activation='relu',padding='same')(entree)
    result = layers.BatchNormalization()(result)
    result=resblock(result,3,nbr)
    result1=resblock(result,3,nbr)
    result=layers.MaxPool2D()(result1)

    #level-1
    result=layers.Conv2D(2*nbr,3,activation='relu',padding='same')(result)
    result=layers.BatchNormalization()(result)
    result = resblock(result,3,2*nbr)
    result2 = resblock(result, 3, 2 * nbr)
    result=layers.MaxPool2D()(result2)

    #Level -2
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result=resblock(result,3,4*nbr)
    result3 = resblock(result, 3, 4 * nbr)
    result=layers.MaxPool2D()(result3)

    #Level-3
    result = layers.Conv2D(8 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = resblock(result, 3, 8 * nbr)
    result = resblock(result, 3, 8 * nbr)
    result = layers.Conv2DTranspose(4*nbr,3,strides=(2,2),activation='relu',padding='same')(result)

    #Level -2
    result=layers.Concatenate()([result,result3])
    result = layers.Conv2D(4 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = resblock(result, 3, 4 * nbr)
    result = resblock(result, 3, 4 * nbr)
    result = layers.Conv2DTranspose(2*nbr,3,strides=(2,2),activation='relu',padding='same')(result)


    #Level-1
    result = layers.Concatenate()([result, result2])
    result = layers.Conv2D(2 * nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = resblock(result, 3, 2 * nbr)
    result = resblock(result, 3, 2 * nbr)
    result = layers.Conv2DTranspose(nbr,3,strides=(2,2),activation='relu',padding='same')(result)


    #Level0
    result = layers.Concatenate()([result, result1])
    result = layers.Conv2D(nbr, 3, activation='relu', padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = resblock(result, 3, nbr)
    result = resblock(result, 3, nbr)

    #Output
    sortie = layers.Conv2D(nbr_mask, 1, activation=activation, padding='same')(result)

    model = models.Model(inputs=entree, outputs=sortie)
    return model
    
print(resunetv2().summary())

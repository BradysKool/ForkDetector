import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras import Sequential
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Normalization

imageX = np.load('forkImages.npy')
imageY = np.load('forkY.npy')
print(imageX.shape)
print(imageY.shape)

normalizationLayer = Normalization()
normalizationLayer.adapt(imageX)

model = Sequential([ 
    normalizationLayer,
    Dense(units=100,activation='relu',kernel_regularizer=l2(0.1)),
    Dense(units=15,activation='relu',kernel_regularizer=l2(0.1)),
    Dense(units=1,activation='sigmoid',kernel_regularizer=l2(0.1))
])


model.compile(loss=BinaryCrossentropy(),optimizer=Adam(learning_rate=0.00001,clipnorm=1,clipvalue=1,epsilon=1e-8))



model.fit(imageX,imageY,epochs=5)

print(model.layers[0].weights)

print(imageX.shape[0])
for i in range(imageX.shape[0]):
    prediction = model.predict(np.expand_dims(imageX[i],axis=0))
    cutOf = 0
    if prediction >= 5:
        cutOf=1
    if cutOf != imageY:
        print('Actual: ' + str(imageY[i]) + ', Prediction: ' + str(prediction))

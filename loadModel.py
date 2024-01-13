from keras.models import load_model
import numpy as np

imageX = np.load('forkImages2.npy')
imageY = np.load('forkY2.npy')

model = load_model('forkModel')

model.fit(imageX,imageY,epochs=5)
incorrect = 0

predictons = model.predict(imageX)

for i,pre in enumerate(predictons):
    cutOf = 0
    if pre >= .5:
        cutOf=1
    if cutOf != imageY[i]: 
        print('Actual: ' + str(imageY[i]) + ', Prediction: ' + str(pre))
        incorrect += 1

print('There are ' + str(incorrect) + ' incorrect predictiotns')
model.save('forkModel')
import cv2
import numpy as np
from keras.layers import RandomFlip,RandomRotation
from keras import Sequential

imageX = np.load('forkImages2.npy')
imageY = np.load('forkY2.npy')
print(imageY.shape)

forkList = []
images = []

data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
])

pic = cv2.VideoCapture(0,cv2.CAP_DSHOW)

for i in range(20):
    print('press any key to take picture...')
    input('')
    ret,frame = pic.read()
    if ret:
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        forkTF = input('Is this a fork y/n: ')
        if forkTF == 'y':
            forkTF = 1
        else:
            forkTF = 0
        
        

        for i in range(20):
            agmentedImage = np.expand_dims(grayFrame,0)
            agmentedImage = data_augmentation(agmentedImage)
            agmentedImage = agmentedImage[0].numpy().flatten()
            images.append(agmentedImage)
            forkList.append(forkTF)

        grayFrame = grayFrame.flatten()
        forkList.append(forkTF)
        images.append(grayFrame)


npImages = np.array(images)
forkList = np.array(forkList)
imageX = np.concatenate((imageX,npImages))
imageY = np.concatenate((imageY,forkList))

print(imageX.shape)
print(imageY.shape)
print(imageY)

np.save('forkImages2.npy',imageX)
np.save('forkY2.npy',imageY)

pic.release()

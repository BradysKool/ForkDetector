import numpy as np
from keras.models import load_model
import cv2

model = load_model('forkModel')

pic = cv2.VideoCapture(0,cv2.CAP_DSHOW)

for i in range(5):
    ret,frame = pic.read()
    ret,frame = pic.read()
    if ret:
        cv2.imshow('Fork',frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        imageArray = np.array(frame.flatten())
        imageArray = np.expand_dims(imageArray,0)
        predict = model.predict(imageArray)
        forkPresence = 'I dont see a fork :('
        if predict > .5:
            forkPresence = 'I see a fork :)'
        print(forkPresence)
        print(predict)
        print("Press enter to take new picture")
        cv2.waitKey(0)
        
cv2.destroyAllWindows()

# ForkDetector
Detects if there is a fork or not in a picture.

In the colletionFork.py file, I collected the data that I needed while using data augmentation to speed up the process.

In trainFork.py this is where I initially trained the model to test to see how it works and if it was a good model or not.

Then I created the loadModel.py script to further train the model when I found the model that worked for me. The model ended up having three hidden layers that used the 'relu' activation.

lastly, I created the camFork.py to predict pictures that I was taking and that were not in my dataset. Overall it works fairly well predicting about 70 percent of the pictures correctly.

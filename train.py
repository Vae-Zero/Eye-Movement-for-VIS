import numpy as np
import os
import cv2
from tqdm import tqdm
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

# width, height = 2560, 1440
width, height = 1920, 1080
root = "image/"

filepaths = os.listdir(root)
X, Y = [], []
for filepath in filepaths:
  x, y, _ = filepath.split('_')
  x = float(x) / width
  y = float(y) / height
  X.append(cv2.imread(root + filepath))
  Y.append([x, y])
X = np.array(X) / 255.0
Y = np.array(Y)
print (X.shape, Y.shape)

model = Sequential()
model.add(Conv2D(32, 3, 2, activation = 'relu', input_shape = (12, 44, 3)))
model.add(Conv2D(64, 2, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.summary()

epochs = 400
for epoch in tqdm(range(epochs)):
  model.fit(X, Y, batch_size = 32)
  
model.save("eye_track_model")
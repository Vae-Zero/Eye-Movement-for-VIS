# from matplotlib.font_manager import _Weight
import numpy as np
import pyautogui
import os
import cv2
from tensorflow import keras
import time

pyautogui.FAILSAFE = False

cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

def normalize(x):
  minn, maxx = x.min(), x.max()
  return (x - minn) / (maxx - minn)
  
def scan(image_size=(32, 32)):
  _, frame = video_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  boxes = cascade.detectMultiScale(gray, 1.3, 10)
  if len(boxes) == 2:
    eyes = []
    for box in boxes:
      x, y, w, h = box
      eye = frame[y:y + h, x:x + w]
      eye = cv2.resize(eye, image_size)
      eye = normalize(eye)
      eye = eye[10:-10, 5:-5]
      eyes.append(eye)
    return (np.hstack(eyes) * 255).astype(np.uint8)
  else:
    return None

width, height = pyautogui.size()
model = keras.models.load_model("eye_track_model_saved")


# init kalman filter object
process_noise = 0.03
measure_noise = 500

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * process_noise

kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * measure_noise

#kalman.measurementNoiseCov = np.array([[90, 0],
#                                       [0, 50]], np.float32)

prediction = np.zeros((2, 1), np.float32)

# initialize drawing
img = np.zeros((height, width, 3),np.uint8)
img = 255 - img

before_kalman_color = (0,255,0)
before_kalman_thickness = 2
after_kalman_color = (255,0,0)
after_kalman_thickness = 3
x_old, y_old = -1, -1
prediction_old = -np.ones((2, 1), np.float32)

start_time = time.time()

while True:
  # print("...")
  eyes = scan()
  if not eyes is None:
    eyes = np.expand_dims(eyes / 255.0, axis = 0)
    x, y = model.predict(eyes)[0]
    kalman.correct(np.array([x,y]))
    prediction = kalman.predict()
    pyautogui.moveTo(prediction[0] * width, prediction[1] * height, duration = 0.1)

    if x_old >= 0: # draw
      print("draw line...")
      cv2.line(img, (int(x_old * width), int(y_old * height)), (int(x * width), int(y * height)), before_kalman_color, before_kalman_thickness)
      cv2.line(img, (int(prediction_old[0][0] * width), int(prediction_old[1][0] * height)), (int(prediction[0][0] * width), int(prediction[1][0] * height)), after_kalman_color, after_kalman_thickness)
    x_old, y_old = x, y
    prediction_old = prediction

    current_time = time.time()
    print(current_time-start_time)
    if current_time - start_time > 30:
      cv2.imshow("track", img)
      cv2.waitKey(0)
      cv2.imwrite("track.png", img)
      break
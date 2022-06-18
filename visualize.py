import tkinter as tk
import tkinter.filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import pyautogui
import os
from tensorflow import keras
import time   
import math

pyautogui.FAILSAFE = False

cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)
  
def scan(image_size = (32, 32)):
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
    
width, height = 1920, 1080
#width, height = pyautogui.size()
model = keras.models.load_model("eye_track_model_saved")


# init kalman filter object
process_noise = 0.03
measure_noise = 200

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

prediction = np.zeros((2, 1), np.float32)

# fisheye distortion
f = 200
global distorted 

def distort(undistorted, x0, y0):
    # img = np.copy(undistorted)
    img = np.zeros(undistorted.shape, np.uint8)
    x_min = img.shape[1]
    x_max = 0
    y_min = img.shape[0]
    y_max = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            dx = j - x0
            dy = i - y0
            rc = math.sqrt(dx * dx + dy * dy)
            theta = math.atan2(rc, f)
            gamma = math.atan2(dy, dx)
            rf = f * theta
            x = int(x0 + rf * math.cos(gamma))
            y = int(y0 + rf * math.sin(gamma))
            img[y,x] = undistorted[i,j]
            x_max = max(x_max, x)
            x_min = min(x_min, x)
            y_max = max(y_max, y)
            y_min = min(y_min, y)
    return img[y_min:y_max, x_min:x_max]
    
    
def choose_file(): # 选择文件
    selectFileName = tk.filedialog.askopenfilename(title = '选择文件')
    e.set(selectFileName)
    global img_show
    img_open = Image.open(e_entry.get())
    img_show = ImageTk.PhotoImage(img_open)
    labelShowImage.config(image = img_show)
    labelShowImage.image = img_show

def show(path):
    img_open = Image.open(path)
    img_show = ImageTk.PhotoImage(img_open)
    labelShowImage.config(image = img_show)
    labelShowImage.image = img_show

def visualize():
    start_time = time.time()
    img = cv2.imread(e_entry.get())
    print(img.shape)
    

    while True:
        # print("...")
        eyes = scan()
        if not eyes is None:
            eyes = np.expand_dims(eyes / 255.0, axis = 0)
            x, y = model.predict(eyes)[0]
            kalman.correct(np.array([x,y]))
            prediction = kalman.predict()
            x0 = prediction[0] * width
            y0 = prediction[1] * height            
            pyautogui.moveTo(x0, y0, duration = 0.1)
            
            # distortion center in condition that the tk window hasn't move
            x0 = int(x0 - 111)
            y0 = int(y0 - 245)
            if (x0 >= 0 and x0 < 300):
                if (y0 >= 0 and y0 < 200):
                    show('1.png')
                elif (y0 >= 200 and y0 < 400):
                    show('2.png')
            elif (x0 >= 300 and x0 < 600):
                if (y0 >= 0 and y0 < 200):
                    show('3.png')
                elif (y0 >= 200 and y0 < 400):
                    show('4.png')


            
        current_time = time.time()    
        #print(current_time-start_time)
        if current_time - start_time > 20:
            break


win = tk.Tk()
win.geometry('1920x1080')
win.configure(bg = 'white')
win.title('EyeTrackVIS')
win.resizable(False, False)


label = tk.Label(text = "Path :", bg = 'white', justify = 'left')
label.place(x = 100, y = 50)
global e
e = tk.StringVar()
e_entry = tk.Entry(win, width = 40, textvariable = e)
e_entry.place(x = 200, y = 50)
   

sumbit_btn = tk.Button(win, text = "Choose",  bg = 'white', command = choose_file)
sumbit_btn.place(x = 200, y = 100)
visualize_btn = tk.Button(win, text = "Visiualize", bg = 'white', command = visualize)
visualize_btn.place(x = 500, y = 100)

labelShowImage = tk.Label(width = 600, height = 450)
labelShowImage.place(x = 100, y = 200)
  
win.mainloop()
     
  
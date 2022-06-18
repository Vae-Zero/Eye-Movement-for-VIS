import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

f = 200
def distort(undistorted, x0, y0):
    # img = np.copy(undistorted)
    img = np.zeros(undistorted.shape)
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
    
img = cv2.imread('insurance_form.jpg', cv2.IMREAD_GRAYSCALE)
distorted = cv2.resize(distort(img, 350, 300), (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
plt.figure(figsize = (12,12))
plt.imshow(distorted, cmap = "gray")
plt.show()
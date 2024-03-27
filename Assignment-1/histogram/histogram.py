import cv2
import numpy as np
from matplotlib import pyplot as plt

image_name = "lena.png"
image_o = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

if image_o is None :
    print("Image not loaded.")
    exit()

image_e = cv2.equalizeHist(image_o)

hist_o = cv2.calcHist([image_o], [0], None, [256], [0, 256])
print(hist_o)
hist_e = cv2.calcHist([image_e], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 8))
plt.subplot(221), plt.imshow(image_o, 'gray'), plt.title('original')
plt.subplot(222), plt.imshow(image_e, 'gray'), plt.title('equalized')
plt.subplot(223), plt.plot(hist_o, range=(0, 255), bins=255), plt.title('equalized histogram')
plt.subplot(224), plt.plot(hist_e, range=(0, 255), bins=255), plt.title('original histogram')
plt.show()
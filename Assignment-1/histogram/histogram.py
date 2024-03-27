import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

image_name = "lena_gray.png"
image_o = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

if image_o is None :
    print("Image not loaded.")
    exit()

hist = cv2.calcHist([image_o], [0], None, [256], [0, 256])

cdf = copy.deepcopy(hist)
for i in range(1, 256) :
    cdf[i] = cdf[i - 1] + cdf[i]
cdf = np.round((cdf * 255) / (image_o.shape[0] * image_o.shape[1]))

image_e = copy.deepcopy(image_o)
for i in range(0, image_e.shape[0]) :
    for j in range(0, image_e.shape[1]) :
        image_e[i][j] = cdf[image_e[i][j]]
hist_e = cv2.calcHist([image_e], [0], None, [256], [0, 256])
cdf_e = copy.deepcopy(hist_e)
for i in range(1, 256) :
    cdf_e[i] = cdf_e[i - 1] + cdf_e[i]
cdf_e = np.round((cdf_e * 255) / (image_e.shape[0] * image_e.shape[1]))

plt.figure(figsize=(20, 12)), plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.subplot(321), plt.imshow(image_o, 'gray'), plt.title('original')
plt.subplot(323), plt.bar(range(256), hist.ravel(), color='red'), plt.title('original histogram')
plt.subplot(325), plt.plot(cdf), plt.title('original cdf')
plt.subplot(322), plt.imshow(image_e, 'gray'), plt.title('equalized')
plt.subplot(324), plt.bar(range(256), hist_e.ravel()), plt.title('equalized histogram')
plt.subplot(326), plt.plot(cdf_e), plt.title('equalized cdf')
plt.show()
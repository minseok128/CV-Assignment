import cv2
import numpy as np
import matplotlib

image_name = "lena.png"
image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
if image is not None :
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(image_name)
else :
    print("Image not loaded.")
import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

# def city_block(image, i, j, forward):
# 	i1, j1 = i - forward, j
# 	i2, j2 = i, j - forward
# 	if i1 < 0 or i1 >= HEIGHT or j1 < 0 or j1 >= WIDTH:
# 		i1, j1 = i, j
# 	if i2 < 0 or i2 >= HEIGHT or j2 < 0 or j2 >= WIDTH:
# 		i2, j2 = i, j
# 	return min(image[i][j], image[i1][j1] + 1, image[i2][j2] + 1)
def city_block(image, i, j, forward):
	i1, j1 = i - forward, j
	i2, j2 = i - forward, j - forward
	i3, j3 = i, j - forward
	i4, j4 = i + forward, j - forward
	if i1 < 0 or i1 >= HEIGHT or j1 < 0 or j1 >= WIDTH:
		i1, j1 = i, j
	if i2 < 0 or i2 >= HEIGHT or j2 < 0 or j2 >= WIDTH:
		i2, j2 = i, j
	if i3 < 0 or i3 >= HEIGHT or j3 < 0 or j3 >= WIDTH:
		i3, j3 = i, j
	if i4 < 0 or i4 >= HEIGHT or j4 < 0 or j4 >= WIDTH:
		i4, j4 = i, j
	return min(image[i][j], image[i1][j1] + 1, image[i2][j2] + 2, image[i3][j3] + 1, image[i4][j4] + 2)

# 이미지 파일 이름 설정
image_name = "distance.png"
# 이미지를 그레이스케일로 읽기
image_o = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
HEIGHT, WIDTH = image_o.shape[0], image_o.shape[1]
print(WIDTH, HEIGHT)

image_test = copy.deepcopy(image_o)
for i in range(0, HEIGHT):
	for j in range(0, WIDTH):
		if image_o[i][j] > 0:
			image_test[i][j] = 255
		else:
			image_test[i][j] = 0

for i in range(0, HEIGHT):
	for j in range(0, WIDTH):
		image_test[i][j] = city_block(image_test, i, j, 1)
for i in range(HEIGHT - 1, -1, -1):
	for j in range(WIDTH - 1, -1, -1):
		image_test[i][j] = city_block(image_test, i, j, -1)
cv2.normalize(image_test, image_test, 0, 255, cv2.NORM_MINMAX)


# 결과 표시
plt.figure(figsize=(16, 10)), plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.subplot(221), plt.imshow(image_o, 'gray'), plt.title('Original image')
plt.subplot(222), plt.imshow(image_test, 'gray'), plt.title('Original image')
plt.show()


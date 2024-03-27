import cv2
import copy
import numpy as np
import math
from matplotlib import pyplot as plt

ROOT2 = math.sqrt(2)

def euclidean(ij, ij1, ij2, ij3, ij4):
	return min(ij, ij1 + 1.0, ij2 + ROOT2, ij3 + 1.0, ij4 + ROOT2)

def city_block(ij, ij1, ij2, ij3, ij4):
	return min(ij, ij1 + 1.0, ij2 + 2.0, ij3 + 1.0, ij4 + 2.0)

def chess_board(ij, ij1, ij2, ij3, ij4):
	return min(ij, ij1 + 1.0, ij2 + 1.0, ij3 + 1.0, ij4 + 1.0)

def get_value(image, i, j, forward, func):
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
	return func(image[i][j], image[i1][j1], image[i2][j2], image[i3][j3], image[i4][j4])

# 이미지 파일 이름 설정
image_name = "distance-4.png"
# 이미지를 그레이스케일로 읽기
image_o = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
HEIGHT, WIDTH = image_o.shape[0], image_o.shape[1]
print(WIDTH, HEIGHT)

image_euc = np.zeros_like(image_o, dtype='float64')
for j in range(0, WIDTH):
	for i in range(0, HEIGHT):
		if image_o[i][j] == 255:
			image_euc[i][j] = 255.0
		else:
			image_euc[i][j] = 0.0
image_cty = copy.deepcopy(image_euc)
image_chs = copy.deepcopy(image_euc)

for j in range(0, WIDTH):
	for i in range(0, HEIGHT):
		image_euc[i][j] = get_value(image_euc, i, j, 1, euclidean)
		image_cty[i][j] = get_value(image_cty, i, j, 1, city_block)
		image_chs[i][j] = get_value(image_chs, i, j, 1, chess_board)
for j in range(WIDTH - 1, -1, -1):
	for i in range(HEIGHT - 1, -1, -1):
		image_euc[i][j] = get_value(image_euc, i, j, -1, euclidean)
		image_cty[i][j] = get_value(image_cty, i, j, -1, city_block)
		image_chs[i][j] = get_value(image_chs, i, j, -1, chess_board)


cv2.normalize(image_euc, image_euc, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(image_cty, image_cty, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(image_chs, image_chs, 0, 255, cv2.NORM_MINMAX)

# 결과 표시
plt.figure(figsize=(16, 10)), plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.subplot(221), plt.imshow(image_o, 'gray'), plt.title('Original image')
plt.subplot(222), plt.imshow(image_euc, 'gray'), plt.title('Euclidean distance')
plt.subplot(223), plt.imshow(image_cty, 'gray'), plt.title('City block distance')
plt.subplot(224), plt.imshow(image_chs, 'gray'), plt.title('Chess board distance')
plt.show()


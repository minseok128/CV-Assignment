import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

# 루트 2 값 (대각선 거리 계산용)
ROOT2 = 2**0.5

# 유클리디안 거리 변환 함수
def euclidean(ij, ij1, ij2, ij3, ij4):
    # 현재 픽셀과 이웃 픽셀들 사이의 유클리디안 거리를 계산
    return min(ij, ij1 + 1.0, ij2 + ROOT2, ij3 + 1.0, ij4 + ROOT2)

# 시티 블록(맨해튼) 거리 변환 함수
def city_block(ij, ij1, ij2, ij3, ij4):
    # 현재 픽셀과 이웃 픽셀들 사이의 시티 블록 거리를 계산
    return min(ij, ij1 + 1.0, ij2 + 2.0, ij3 + 1.0, ij4 + 2.0)

# 체스 보드 거리 변환 함수
def chess_board(ij, ij1, ij2, ij3, ij4):
    # 현재 픽셀과 이웃 픽셀들 사이의 체스 보드 거리를 계산
    return min(ij, ij1 + 1.0, ij2 + 1.0, ij3 + 1.0, ij4 + 1.0)

# 특정 거리 측정 함수를 사용하여 이미지의 픽셀 값 갱신
def get_value(image, i, j, forward, func):
    # 주변 픽셀 위치 계산 및 경계 검사
    i1, j1 = i - forward, j
    i2, j2 = i - forward, j - forward
    i3, j3 = i, j - forward
    i4, j4 = i + forward, j - forward
    # 경계 외부에 위치한 픽셀 처리
    if i1 < 0 or i1 >= HEIGHT or j1 < 0 or j1 >= WIDTH:
        i1, j1 = i, j
    if i2 < 0 or i2 >= HEIGHT or j2 < 0 or j2 >= WIDTH:
        i2, j2 = i, j
    if i3 < 0 or i3 >= HEIGHT or j3 < 0 or j3 >= WIDTH:
        i3, j3 = i, j
    if i4 < 0 or i4 >= HEIGHT or j4 < 0 or j4 >= WIDTH:
        i4, j4 = i, j
    # 주변 픽셀 값과 현재 픽셀 값에 따른 거리 계산
    return func(image[i][j], image[i1][j1], image[i2][j2], image[i3][j3], image[i4][j4])

# 이미지 로드 및 초기 설정
image_name = "distance-5.png"
image_o = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 이미지 읽기
HEIGHT, WIDTH = image_o.shape[0], image_o.shape[1]  # 이미지의 너비와 높이
print(WIDTH, HEIGHT)

image_euc = np.zeros_like(image_o, dtype='float64') # 유클리디안 거리 이미지
image_euc[image_o == 255] = 255.0
image_euc[image_o != 255] = 0.0
image_cty = copy.deepcopy(image_euc) # 시티 블록 거리 이미지
image_chs = copy.deepcopy(image_euc) # 체스 보드 거리 이미지

# 각 거리 측정 기준에 따른 거리 변환 계산
for j in range(0, WIDTH):
    for i in range(0, HEIGHT):
        # 각 픽셀에 대해 유클리디안, 시티 블록, 체스 보드 거리 적용
        image_euc[i][j] = get_value(image_euc, i, j, 1, euclidean)
        image_cty[i][j] = get_value(image_cty, i, j, 1, city_block)
        image_chs[i][j] = get_value(image_chs, i, j, 1, chess_board)

# 각 픽셀에 대해 거리 변환을 역방향으로 재적용
for j in range(WIDTH - 1, -1, -1):
    for i in range(HEIGHT - 1, -1, -1):
        image_euc[i][j] = get_value(image_euc, i, j, -1, euclidean)
        image_cty[i][j] = get_value(image_cty, i, j, -1, city_block)
        image_chs[i][j] = get_value(image_chs, i, j, -1, chess_board)

# 거리 변환 결과를 0부터 255 사이의 범위로 정규화
cv2.normalize(image_euc, image_euc, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(image_cty, image_cty, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(image_chs, image_chs, 0, 255, cv2.NORM_MINMAX)

# 결과 시각화
plt.figure(figsize=(16, 10)), plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.subplot(221), plt.imshow(image_o, 'gray'), plt.title('Original image')  # 원본 이미지 출력
plt.subplot(222), plt.imshow(image_euc, 'gray'), plt.title('Euclidean distance')  # 유클리디안 거리 변환 결과 출력
plt.subplot(223), plt.imshow(image_cty, 'gray'), plt.title('City block distance')  # 시티 블록 거리 변환 결과 출력
plt.subplot(224), plt.imshow(image_chs, 'gray'), plt.title('Chess board distance')  # 체스 보드 거리 변환 결과 출력
plt.show()

import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

L = 256

# 이미지 파일 이름 설정
image_name = "lena_gray.png"
# 이미지를 그레이스케일로 읽기
image_o = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

# 이미지가 정상적으로 로드되었는지 확인
if image_o is None:
    print("Image not loaded.")
    exit()

# 원본 이미지의 히스토그램 계산
hist = cv2.calcHist([image_o], [0], None, [L], [0, L])

# 원본 이미지의 CDF(누적 분포 함수) 계산
cdf = copy.deepcopy(hist)
for i in range(1, L):
    cdf[i] = cdf[i - 1] + cdf[i]

# CDF 정규화 및 정수 변환
cdf = np.round((cdf * (L - 1)) / (image_o.shape[0] * image_o.shape[1])).astype('int')

# Equalized 이미지 생성
image_e = copy.deepcopy(image_o)
for i in range(image_e.shape[0]):
    for j in range(image_e.shape[1]):
        image_e[i][j] = cdf[int(image_e[i][j])]

# Equalized 이미지의 히스토그램 계산
hist_e = cv2.calcHist([image_e], [0], None, [L], [0, L])

# Equalized 이미지의 CDF 계산
cdf_e = copy.deepcopy(hist_e)
for i in range(1, L):
    cdf_e[i] = cdf_e[i - 1] + cdf_e[i]

# Equalized 이미지의 CDF 정규화
cdf_e = np.round((cdf_e * (L - 1)) / (image_e.shape[0] * image_e.shape[1]))

# 결과 표시
plt.figure(figsize=(16, 10)), plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.subplot(321), plt.imshow(image_o, 'gray'), plt.title('Original image')
plt.subplot(322), plt.imshow(image_e, 'gray'), plt.title('Equalized image')

plt.subplot(325), plt.plot(cdf), plt.title('Original cdf * (L - 1)')
plt.subplot(326), plt.plot(cdf_e), plt.title('Equalized cdf * (L - 1)')

plt.subplot(323), plt.bar(range(L), hist.ravel(), color='red'), plt.title('Original histogram')
plt.subplot(324), plt.bar(range(L), hist_e.ravel()), plt.title('Equalized histogram')
plt.show()

import cv2 as cv  # OpenCV 라이브러리를 불러옴
import numpy as np  # 수치 연산용 NumPy를 불러옴
import matplotlib.pyplot as plt  # 시각화용 Matplotlib를 불러옴
img = cv.imread('edgeDetectionImage.jpg')  # 입력 이미지를 파일에서 읽어옴
if img is None:  # 이미지 로드 실패 여부를 확인함
	raise FileNotFoundError('edgeDetectionImage.jpg 파일을 찾을 수 없습니다.')  # 파일이 없으면 오류를 발생시킴
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 컬러 이미지를 그레이스케일로 변환함
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # x축 방향 소벨 에지를 계산함
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # y축 방향 소벨 에지를 계산함
edge_mag = cv.magnitude(sobel_x, sobel_y)  # x/y 에지를 합쳐 에지 강도를 계산함
edge_uint8 = cv.convertScaleAbs(edge_mag)  # 에지 강도 영상을 uint8로 변환함
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Matplotlib 표시를 위해 BGR을 RGB로 변환함
plt.figure(figsize=(12, 5))  # 시각화용 Figure 크기를 설정함
plt.subplot(1, 2, 1)  # 첫 번째 서브플롯을 선택함
plt.imshow(img_rgb)  # 원본 이미지를 표시함
plt.title('Original Image')  # 첫 번째 이미지 제목을 설정함
plt.axis('off')  # 축 표시를 숨김
plt.subplot(1, 2, 2)  # 두 번째 서브플롯을 선택함
plt.imshow(edge_uint8, cmap='gray')  # 에지 강도 이미지를 흑백으로 표시함
plt.title('Sobel Edge Magnitude')  # 두 번째 이미지 제목을 설정함
plt.axis('off')  # 축 표시를 숨김
plt.tight_layout()  # 레이아웃이 겹치지 않게 자동 조정함
plt.show()  # 화면에 시각화 결과를 출력함

import cv2 as cv  # OpenCV 라이브러리를 불러옴
import numpy as np  # 수치 연산용 NumPy를 불러옴
import matplotlib.pyplot as plt  # 시각화용 Matplotlib를 불러옴
img = cv.imread('coffee cup.JPG')  # 입력 이미지를 파일에서 읽어옴
if img is None:  # 이미지 로드 실패 여부를 확인함
	raise FileNotFoundError('coffee cup.JPG 파일을 찾을 수 없습니다.')  # 파일이 없으면 오류를 발생시킴
preview = img.copy()  # ROI 선택 창에 보여줄 복사 이미지를 만듦
rect = cv.selectROI('Select ROI for GrabCut', preview, showCrosshair=True, fromCenter=False)  # 사용자가 사각형 ROI를 대화식으로 선택함
cv.destroyWindow('Select ROI for GrabCut')  # ROI 선택 창을 닫음
x, y, w, h = rect  # 선택된 사각형 좌표를 분해함
if w == 0 or h == 0:  # ROI를 선택하지 않았는지 확인함
	h_img, w_img = img.shape[:2]  # 원본 이미지 높이와 너비를 가져옴
	x = int(w_img * 0.2)  # 기본 ROI의 시작 x를 계산함
	y = int(h_img * 0.2)  # 기본 ROI의 시작 y를 계산함
	w = int(w_img * 0.6)  # 기본 ROI의 너비를 계산함
	h = int(h_img * 0.6)  # 기본 ROI의 높이를 계산함
	rect = (x, y, w, h)  # 계산한 기본 ROI를 사각형으로 설정함
mask = np.zeros(img.shape[:2], np.uint8)  # GrabCut용 마스크를 0으로 초기화함
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델 배열을 초기화함
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델 배열을 초기화함
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)  # 선택한 사각형 기반으로 GrabCut 분할을 수행함
binary_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')  # 배경/가능배경은 0, 전경/가능전경은 1로 변환함
result = img * binary_mask[:, :, np.newaxis]  # 이진 마스크를 원본에 곱해 배경을 제거함
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 원본 표시를 위해 RGB로 변환함
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)  # 결과 표시를 위해 RGB로 변환함
mask_vis = (binary_mask * 255).astype(np.uint8)  # 마스크를 시각화용 0/255 이미지로 변환함
plt.figure(figsize=(15, 5))  # 시각화용 Figure 크기를 설정함
plt.subplot(1, 3, 1)  # 첫 번째 서브플롯을 선택함
plt.imshow(img_rgb)  # 원본 이미지를 표시함
plt.title('Original Image')  # 첫 번째 이미지 제목을 설정함
plt.axis('off')  # 축 표시를 숨김
plt.subplot(1, 3, 2)  # 두 번째 서브플롯을 선택함
plt.imshow(mask_vis, cmap='gray')  # 마스크 이미지를 흑백으로 표시함
plt.title('GrabCut Mask')  # 두 번째 이미지 제목을 설정함
plt.axis('off')  # 축 표시를 숨김
plt.subplot(1, 3, 3)  # 세 번째 서브플롯을 선택함
plt.imshow(result_rgb)  # 배경 제거 결과 이미지를 표시함
plt.title('Foreground Extracted')  # 세 번째 이미지 제목을 설정함
plt.axis('off')  # 축 표시를 숨김
plt.tight_layout()  # 레이아웃이 겹치지 않게 자동 조정함
plt.show()  # 화면에 시각화 결과를 출력함

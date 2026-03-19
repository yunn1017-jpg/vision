import cv2 as cv  # OpenCV 라이브러리를 불러옴
import numpy as np  # 수치 연산용 NumPy를 불러옴
import matplotlib.pyplot as plt  # 시각화용 Matplotlib를 불러옴
img = cv.imread('dabo.jpg')  # 입력 이미지를 파일에서 읽어옴
if img is None:  # 이미지 로드 실패 여부를 확인함
	raise FileNotFoundError('dabo.jpg 파일을 찾을 수 없습니다.')  # 파일이 없으면 오류를 발생시킴
original_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 원본 표시용으로 RGB 변환본을 만듦
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 캐니 연산을 위해 그레이스케일로 변환함
edges = cv.Canny(gray, 100, 200)  # 임계값 100/200으로 캐니 에지 맵을 생성함
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)  # 확률적 허프 변환으로 직선을 검출함
line_img = img.copy()  # 직선을 그릴 복사 이미지를 생성함
if lines is not None:  # 검출된 직선이 있는지 확인함
	for line in lines:  # 검출된 각 직선에 대해 반복함
		x1, y1, x2, y2 = line[0]  # 직선의 시작점과 끝점 좌표를 꺼냄
		cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 두께 2로 직선을 그림
line_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)  # Matplotlib 표시를 위해 RGB로 변환함
plt.figure(figsize=(12, 5))  # 시각화용 Figure 크기를 설정함
plt.subplot(1, 2, 1)  # 첫 번째 서브플롯을 선택함
plt.imshow(original_rgb)  # 원본 이미지를 표시함
plt.title('Original Image')  # 첫 번째 이미지 제목을 설정함
plt.axis('off')  # 축 표시를 숨김
plt.subplot(1, 2, 2)  # 두 번째 서브플롯을 선택함
plt.imshow(line_rgb)  # 직선이 그려진 결과 이미지를 표시함
plt.title('Canny + HoughLinesP Result')  # 두 번째 이미지 제목을 설정함
plt.axis('off')  # 축 표시를 숨김
plt.tight_layout()  # 레이아웃이 겹치지 않게 자동 조정함
plt.show()  # 화면에 시각화 결과를 출력함

import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # Numpy 라이브러리 임포트

img = cv.imread('soccer.jpg')# 이미지 불러오기

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)# 그레이스케일 변환

gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)# gray를 BGR로 변환 (hstack 맞추기 위해)

combined = np.hstack((img, gray_bgr))# 이미지 가로 연결

cv.imshow("Original | Grayscale", combined)# 출력

cv.waitKey(0)# 아무 키 누르면 종료
cv.destroyAllWindows() # 루프가 종료되면 모든 창을 닫고 프로그램 종료

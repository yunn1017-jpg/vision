import cv2          # OpenCV 라이브러리 임포트
import numpy as np  # 수치 계산용 numpy 임포트

# -----------------------------
# 1. 이미지 불러오기
# -----------------------------
img = cv2.imread("images/rose.png")  # 원본 이미지 파일 읽기
if img is None:  # 이미지 로드 실패 시
    raise FileNotFoundError("images/rose.png 파일을 찾을 수 없습니다.")  # 에러 발생

h, w = img.shape[:2]  # 이미지의 높이(h), 너비(w) 가져오기

# -----------------------------
# 2. 회전(Rotation) + 크기 조절(Scaling)
# -----------------------------
angle = 30        # 회전 각도 30도
scale = 0.8       # 크기를 0.8배로 축소

center = (w // 2, h // 2)  # 이미지 중심 좌표 계산
M_rotate = cv2.getRotationMatrix2D(center, angle, scale)  # 중심 기준 회전+크기조절 변환 행렬 생성 (2x3)

rotated_scaled = cv2.warpAffine(img, M_rotate, (w, h))  # 변환 행렬을 적용하여 회전+크기조절 수행

# -----------------------------
# 3. 이동(Translation)
# -----------------------------
tx, ty = 80, -40  # x방향 +80px, y방향 -40px 이동량 설정

M_translate = np.float32([[1, 0, tx],   # 이동 변환 행렬 생성 (2x3)
                           [0, 1, ty]])  # [[1,0,tx],[0,1,ty]] 형태

transformed = cv2.warpAffine(rotated_scaled, M_translate, (w, h))  # 회전+크기조절된 이미지에 이동 변환 적용

# -----------------------------
# 4. 시각화: 원본 vs 최종 변환 결과
# -----------------------------
combined = np.hstack([img, transformed])  # 원본과 변환 결과를 좌우로 이어 붙이기

max_width = 1200  # 표시할 최대 너비 설정
ch, cw = combined.shape[:2]  # 합친 이미지의 높이, 너비 가져오기
if cw > max_width:  # 최대 너비 초과 시
    ratio = max_width / cw  # 축소 비율 계산
    combined = cv2.resize(combined, (max_width, int(ch * ratio)))  # 비율에 맞게 리사이즈

cv2.imshow("Original (Left) vs Rotated + Scaled + Translated (Right)", combined)  # 비교 이미지 화면에 표시
print("원본(좌)과 변환 결과(우)를 비교합니다. 아무 키나 누르면 종료합니다.")  # 안내 메시지 출력
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 창 닫기

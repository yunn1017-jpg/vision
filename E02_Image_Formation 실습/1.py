import cv2          # OpenCV 라이브러리 임포트
import numpy as np  # 수치 계산용 numpy 임포트
import glob         # 파일 경로 패턴 매칭용 glob 임포트

CHECKERBOARD = (9, 6)  # 체크보드 내부 코너 개수 (가로 9, 세로 6)

square_size = 25.0  # 체크보드 한 칸의 실제 크기 (mm)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 코너 정밀화 종료 조건 (정밀도 or 최대 반복)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)  # 체크보드 코너의 3D 좌표를 담을 배열 생성 (54개 점, z=0)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  # x, y 좌표를 격자 형태로 채움
objp *= square_size  # 실제 크기(25mm)를 곱하여 실세계 좌표로 변환

objpoints = []  # 모든 이미지의 3D 실세계 좌표를 저장할 리스트
imgpoints = []  # 모든 이미지의 2D 이미지 좌표를 저장할 리스트

images = glob.glob("images/calibration_images/left*.jpg")  # 캘리브레이션용 체크보드 이미지 파일 목록 가져오기

img_size = None  # 이미지 크기 저장용 변수 초기화

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:  # 각 이미지 파일에 대해 반복
    img = cv2.imread(fname)  # 이미지 파일 읽기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

    if img_size is None:  # 첫 이미지에서 크기 저장
        img_size = (gray.shape[1], gray.shape[0])  # (너비, 높이) 형태로 저장

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)  # 체크보드 코너 검출 시도

    if ret:  # 코너 검출 성공 시
        objpoints.append(objp)  # 해당 이미지의 3D 좌표 추가
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 서브픽셀 정밀도로 코너 위치 보정
        imgpoints.append(corners2)  # 보정된 2D 코너 좌표 추가
        print(f"[OK] {fname} - 코너 검출 성공")  # 성공 메시지 출력
    else:  # 코너 검출 실패 시
        print(f"[SKIP] {fname} - 코너 검출 실패")  # 실패 메시지 출력

print(f"\n총 {len(imgpoints)}장의 이미지에서 코너 검출 완료\n")  # 검출 완료된 이미지 수 출력

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)  # 3D-2D 매핑으로 카메라 내부 파라미터 계산

print("Camera Matrix K:")  # 카메라 내부 행렬 출력 안내
print(K)  # 3x3 카메라 내부 행렬 (초점거리, 주점 포함) 출력

print("\nDistortion Coefficients:")  # 왜곡 계수 출력 안내
print(dist)  # 렌즈 왜곡 계수 (k1, k2, p1, p2, k3) 출력

print(f"\nRe-projection Error (RMS): {ret:.4f}")  # 재투영 오차 출력 (낮을수록 정확)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
sample_img = cv2.imread(images[0])  # 첫 번째 이미지를 샘플로 읽기
h, w = sample_img.shape[:2]  # 이미지의 높이, 너비 가져오기

new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))  # 왜곡 보정 후 최적 카메라 행렬과 유효 ROI 계산

undistorted = cv2.undistort(sample_img, K, dist, None, new_K)  # 왜곡 보정 적용하여 이미지 펴기

x, y, w_roi, h_roi = roi  # 유효 영역(ROI) 좌표 추출
if w_roi > 0 and h_roi > 0:  # 유효 ROI가 존재하면
    undistorted = undistorted[y:y + h_roi, x:x + w_roi]  # 보정 이미지를 ROI로 크롭
    sample_img_resized = cv2.resize(sample_img, (w_roi, h_roi))  # 원본도 같은 크기로 리사이즈
else:  # ROI가 없으면
    sample_img_resized = sample_img  # 원본 그대로 사용

combined = np.hstack([sample_img_resized, undistorted])  # 원본과 보정 이미지를 좌우로 이어 붙이기

cv2.imshow("Original (Left) vs Undistorted (Right)", combined)  # 비교 이미지 화면에 표시
print("\n원본(좌)과 왜곡 보정(우) 이미지를 비교합니다. 아무 키나 누르면 종료합니다.")  # 안내 메시지 출력
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 창 닫기

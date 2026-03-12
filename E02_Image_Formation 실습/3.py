import cv2          # OpenCV 라이브러리 임포트
import numpy as np  # 수치 계산용 numpy 임포트
from pathlib import Path  # 경로 처리용 pathlib 임포트

output_dir = Path("./outputs")  # 출력 폴더 경로 설정
output_dir.mkdir(parents=True, exist_ok=True)  # 출력 폴더가 없으면 생성

# -----------------------------
# 좌/우 이미지 불러오기
# -----------------------------
left_color = cv2.imread("images/left.png")   # 스테레오 왼쪽 이미지 읽기
right_color = cv2.imread("images/right.png")  # 스테레오 오른쪽 이미지 읽기

if left_color is None or right_color is None:  # 이미지 로드 실패 시
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")  # 에러 발생

f = 700.0    # 카메라 초점 거리 (픽셀 단위)
B = 0.12     # 두 카메라 사이 거리, 베이스라인 (미터)

rois = {  # 관심 영역(ROI) 설정 (x, y, 너비, 높이)
    "Painting": (55, 50, 130, 110),   # 그림 영역
    "Frog": (90, 265, 230, 95),       # 개구리 영역
    "Teddy": (310, 35, 115, 90)       # 곰인형 영역
}

left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)   # 왼쪽 이미지를 그레이스케일로 변환
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)  # 오른쪽 이미지를 그레이스케일로 변환

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)  # StereoBM 매칭 객체 생성 (탐색범위 64, 블록크기 15)
disparity_raw = stereo.compute(left_gray, right_gray)  # 좌우 이미지로 disparity 계산 (16배 스케일 정수)

disparity = disparity_raw.astype(np.float32) / 16.0  # 16배 스케일을 나눠 실제 disparity 값으로 변환

# -----------------------------
# 2. Depth 계산 (Z = fB / d)
# Disparity > 0인 픽셀만 필터링
# -----------------------------
valid_mask = disparity > 0  # 유효한 disparity (양수)인 픽셀 마스크 생성
depth_map = np.zeros_like(disparity)  # depth 맵을 0으로 초기화
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # Z = fB/d 공식으로 깊이 계산

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}  # 결과 저장용 딕셔너리
for name, (x, y, w, h) in rois.items():  # 각 ROI에 대해 반복
    roi_disp = disparity[y:y + h, x:x + w]  # ROI 영역의 disparity 추출
    roi_depth = depth_map[y:y + h, x:x + w]  # ROI 영역의 depth 추출
    roi_valid = roi_disp > 0  # ROI 내 유효한 픽셀 마스크

    if np.any(roi_valid):  # 유효한 픽셀이 있으면
        mean_disp = np.mean(roi_disp[roi_valid])   # 유효 픽셀의 평균 disparity 계산
        mean_depth = np.mean(roi_depth[roi_valid])  # 유효 픽셀의 평균 depth 계산
    else:  # 유효 픽셀이 없으면
        mean_disp = 0.0   # 평균 disparity를 0으로 설정
        mean_depth = 0.0  # 평균 depth를 0으로 설정

    results[name] = {"mean_disparity": mean_disp, "mean_depth": mean_depth}  # 결과 저장

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=" * 55)  # 구분선 출력
print(f"{'ROI':<12} {'Mean Disparity':>16} {'Mean Depth (m)':>18}")  # 테이블 헤더 출력
print("=" * 55)  # 구분선 출력
for name, vals in results.items():  # 각 ROI 결과 반복 출력
    print(f"{name:<12} {vals['mean_disparity']:>16.2f} {vals['mean_depth']:>18.4f}")  # ROI명, 평균 disparity, 평균 depth 출력
print("=" * 55)  # 구분선 출력

closest = max(results, key=lambda k: results[k]["mean_disparity"])   # disparity가 가장 큰 (가장 가까운) ROI 찾기
farthest = min(results, key=lambda k: results[k]["mean_disparity"])  # disparity가 가장 작은 (가장 먼) ROI 찾기

print(f"\n[해석] Disparity가 클수록 카메라에 가까운 물체입니다.")  # 해석 원리 설명
print(f"  - 가장 가까운 물체: {closest} (disparity = {results[closest]['mean_disparity']:.2f})")  # 가장 가까운 물체 출력
print(f"  - 가장 먼 물체:     {farthest} (disparity = {results[farthest]['mean_disparity']:.2f})")  # 가장 먼 물체 출력

# -----------------------------
# 5. Disparity 시각화
# -----------------------------
disp_tmp = disparity.copy()  # disparity 배열 복사
disp_tmp[disp_tmp <= 0] = np.nan  # 유효하지 않은 값(0 이하)을 NaN으로 설정

if np.all(np.isnan(disp_tmp)):  # 모든 값이 NaN이면
    raise ValueError("유효한 disparity 값이 없습니다.")  # 에러 발생

d_min = np.nanpercentile(disp_tmp, 5)   # 하위 5% 값 (최솟값 기준)
d_max = np.nanpercentile(disp_tmp, 95)  # 상위 95% 값 (최댓값 기준)

if d_max <= d_min:  # 최대-최소 차이가 없으면
    d_max = d_min + 1e-6  # 미세한 차이 추가 (0 나누기 방지)

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)  # 0~1 범위로 정규화
disp_scaled = np.clip(disp_scaled, 0, 1)  # 0~1 범위로 클리핑

disp_vis = np.zeros_like(disparity, dtype=np.uint8)  # 시각화용 8비트 배열 생성
valid_disp = ~np.isnan(disp_tmp)  # 유효한 값 마스크
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)  # 0~255 범위로 변환

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)  # JET 컬러맵 적용 (빨강=가까움, 파랑=멀음)

# -----------------------------
# 6. Depth 시각화
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)  # 시각화용 8비트 배열 생성

if np.any(valid_mask):  # 유효한 depth 값이 있으면
    depth_valid = depth_map[valid_mask]  # 유효한 depth 값 추출

    z_min = np.percentile(depth_valid, 5)   # 하위 5% depth 값
    z_max = np.percentile(depth_valid, 95)  # 상위 95% depth 값

    if z_max <= z_min:  # 최대-최소 차이가 없으면
        z_max = z_min + 1e-6  # 미세한 차이 추가

    depth_scaled = (depth_map - z_min) / (z_max - z_min)  # 0~1 범위로 정규화
    depth_scaled = np.clip(depth_scaled, 0, 1)  # 0~1 범위로 클리핑

    depth_scaled = 1.0 - depth_scaled  # 반전 (가까울수록 높은 값 = 빨강)
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)  # 0~255 범위로 변환

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)  # JET 컬러맵 적용

combined = np.hstack([left_color, disparity_color])  # 원본과 disparity 맵을 좌우로 이어 붙이기

max_width = 1200  # 표시할 최대 너비 설정
ch, cw = combined.shape[:2]  # 합친 이미지의 높이, 너비 가져오기
if cw > max_width:  # 최대 너비 초과 시
    ratio = max_width / cw  # 축소 비율 계산
    combined = cv2.resize(combined, (max_width, int(ch * ratio)))  # 비율에 맞게 리사이즈

cv2.imshow("Original (Left) vs Disparity Map (Right)", combined)  # 비교 이미지 화면에 표시

print("\n시각화 창을 확인하세요. 아무 키나 누르면 종료합니다.")  # 안내 메시지 출력
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 창 닫기

# 결과 이미지 저장
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
print("결과 이미지가 outputs/ 폴더에 저장되었습니다.")

import cv2 as cv  # OpenCV를 불러옵니다.
import numpy as np  # 행렬 연산을 위해 NumPy를 불러옵니다.
import matplotlib.pyplot as plt  # 시각화용 matplotlib을 불러옵니다.

base_path = "img1.jpg"  # 기준 이미지 경로를 지정합니다.
warp_path = "img2.jpg"  # 정합할 이미지 경로를 지정합니다.

base_bgr = cv.imread(base_path)  # 기준 이미지를 읽습니다.
warp_bgr = cv.imread(warp_path)  # 정합할 이미지를 읽습니다.

if base_bgr is None:  # 기준 이미지 로드 실패 여부를 확인합니다.
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {base_path}")  # 파일 없음 예외를 발생시킵니다.
if warp_bgr is None:  # 정합 이미지 로드 실패 여부를 확인합니다.
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {warp_path}")  # 파일 없음 예외를 발생시킵니다.

base_gray = cv.cvtColor(base_bgr, cv.COLOR_BGR2GRAY)  # 기준 이미지를 그레이스케일로 변환합니다.
warp_gray = cv.cvtColor(warp_bgr, cv.COLOR_BGR2GRAY)  # 정합 이미지를 그레이스케일로 변환합니다.

sift = cv.SIFT_create(nfeatures=1200)  # SIFT 객체를 생성합니다.
kp_base, des_base = sift.detectAndCompute(base_gray, None)  # 기준 이미지의 특징점과 기술자를 계산합니다.
kp_warp, des_warp = sift.detectAndCompute(warp_gray, None)  # 정합 이미지의 특징점과 기술자를 계산합니다.

if des_base is None or des_warp is None:  # 기술자 계산 실패 여부를 확인합니다.
    raise RuntimeError("특징점 기술자 계산에 실패했습니다.")  # 계산 실패 예외를 발생시킵니다.

bf = cv.BFMatcher(cv.NORM_L2)  # L2 거리 기반 BFMatcher를 생성합니다.
knn_matches = bf.knnMatch(des_base, des_warp, k=2)  # 각 특징점에 대해 최근접 2개 매칭을 구합니다.

good_matches = []  # 비율 테스트를 통과한 좋은 매칭을 저장할 리스트를 만듭니다.
ratio_thresh = 0.7  # Lowe 비율 테스트 임계값을 설정합니다.
for pair in knn_matches:  # knn 결과를 순회합니다.
    if len(pair) < 2:  # 이웃이 2개 미만인 경우를 제외합니다.
        continue  # 다음 반복으로 이동합니다.
    m, n = pair  # 최근접 매칭과 차최근접 매칭을 분리합니다.
    if m.distance < ratio_thresh * n.distance:  # 비율 테스트를 통과하는지 확인합니다.
        good_matches.append(m)  # 좋은 매칭으로 추가합니다.

if len(good_matches) < 4:  # 호모그래피 계산 최소 조건을 확인합니다.
    raise RuntimeError(f"좋은 매칭 수가 부족합니다: {len(good_matches)}개")  # 매칭 부족 예외를 발생시킵니다.

src_pts = np.float32([kp_warp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 정합 이미지 좌표를 원본 점으로 구성합니다.
dst_pts = np.float32([kp_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 기준 이미지 좌표를 목표 점으로 구성합니다.

H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)  # RANSAC으로 호모그래피를 추정합니다.

if H is None or mask is None:  # 호모그래피 계산 성공 여부를 확인합니다.
    raise RuntimeError("호모그래피 계산에 실패했습니다.")  # 계산 실패 예외를 발생시킵니다.

h1, w1 = base_bgr.shape[:2]  # 기준 이미지의 높이와 너비를 구합니다.
h2, w2 = warp_bgr.shape[:2]  # 정합 이미지의 높이와 너비를 구합니다.
pano_w = w1 + w2  # 파노라마 출력 너비를 계산합니다.
pano_h = max(h1, h2)  # 파노라마 출력 높이를 계산합니다.

warped_bgr = cv.warpPerspective(warp_bgr, H, (pano_w, pano_h))  # 정합 이미지를 기준 좌표계로 변환합니다.
warped_bgr[0:h1, 0:w1] = base_bgr  # 기준 이미지를 좌측 상단에 배치합니다.

inlier_mask = mask.ravel().tolist()  # inlier 마스크를 drawMatches용 리스트로 변환합니다.
match_vis_bgr = cv.drawMatches(  # inlier 중심의 매칭 결과를 시각화합니다.
    base_bgr,  # 기준 이미지를 전달합니다.
    kp_base,  # 기준 이미지 특징점을 전달합니다.
    warp_bgr,  # 정합 이미지를 전달합니다.
    kp_warp,  # 정합 이미지 특징점을 전달합니다.
    good_matches,  # 비율 테스트 통과 매칭을 전달합니다.
    None,  # 결과 이미지를 새로 생성합니다.
    matchesMask=inlier_mask,  # RANSAC inlier만 강조하여 표시합니다.
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  # 단일 점은 표시하지 않습니다.
)  # 매칭 시각화 이미지를 생성합니다.

warped_rgb = cv.cvtColor(warped_bgr, cv.COLOR_BGR2RGB)  # 정합 결과를 RGB로 변환합니다.
match_vis_rgb = cv.cvtColor(match_vis_bgr, cv.COLOR_BGR2RGB)  # 매칭 결과를 RGB로 변환합니다.

plt.figure(figsize=(18, 8))  # 출력 창 크기를 설정합니다.

plt.subplot(1, 2, 1)  # 왼쪽 서브플롯을 선택합니다.
plt.imshow(warped_rgb)  # 정합 결과 이미지를 표시합니다.
plt.title("Warped Image (Homography Alignment)")  # 왼쪽 제목을 설정합니다.
plt.axis("off")  # 축을 숨깁니다.

plt.subplot(1, 2, 2)  # 오른쪽 서브플롯을 선택합니다.
plt.imshow(match_vis_rgb)  # 매칭 결과 이미지를 표시합니다.
plt.title(f"Matching Result - Good: {len(good_matches)}, Inlier: {sum(inlier_mask)}")  # 매칭 통계를 제목에 표시합니다.
plt.axis("off")  # 축을 숨깁니다.

plt.tight_layout()  # 레이아웃을 정리합니다.
plt.show()  # 화면에 결과를 출력합니다.

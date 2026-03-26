import os  # 파일 존재 여부 확인을 위해 os를 불러옵니다.
import cv2 as cv  # OpenCV를 불러옵니다.
import matplotlib.pyplot as plt  # 시각화용 matplotlib을 불러옵니다.

image1_path = "mot_color70.jpg"  # 첫 번째 이미지 경로를 지정합니다.
image2_candidates = ["mot_color80.jpg", "mot_color83.jpg"]  # 두 번째 이미지 후보 목록을 지정합니다.
image2_path = next((p for p in image2_candidates if os.path.exists(p)), None)  # 존재하는 파일을 우선 선택합니다.

if image2_path is None:  # 두 번째 이미지가 없으면 예외 처리합니다.
    raise FileNotFoundError("mot_color80.jpg 또는 mot_color83.jpg 파일이 필요합니다.")  # 파일 없음 예외를 발생시킵니다.

img1_bgr = cv.imread(image1_path)  # 첫 번째 이미지를 읽습니다.
img2_bgr = cv.imread(image2_path)  # 두 번째 이미지를 읽습니다.

if img1_bgr is None:  # 첫 번째 이미지 로드 실패 여부를 확인합니다.
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image1_path}")  # 파일 없음 예외를 발생시킵니다.
if img2_bgr is None:  # 두 번째 이미지 로드 실패 여부를 확인합니다.
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image2_path}")  # 파일 없음 예외를 발생시킵니다.

gray1 = cv.cvtColor(img1_bgr, cv.COLOR_BGR2GRAY)  # 첫 번째 이미지를 그레이스케일로 변환합니다.
gray2 = cv.cvtColor(img2_bgr, cv.COLOR_BGR2GRAY)  # 두 번째 이미지를 그레이스케일로 변환합니다.

sift = cv.SIFT_create(nfeatures=600)  # SIFT 객체를 생성합니다.
kp1, des1 = sift.detectAndCompute(gray1, None)  # 첫 번째 이미지의 특징점과 기술자를 계산합니다.
kp2, des2 = sift.detectAndCompute(gray2, None)  # 두 번째 이미지의 특징점과 기술자를 계산합니다.

if des1 is None or des2 is None:  # 기술자 계산 실패 여부를 확인합니다.
    raise RuntimeError("특징점 기술자 계산에 실패했습니다.")  # 계산 실패 예외를 발생시킵니다.

matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)  # L2 거리 기반 BFMatcher를 생성합니다.
matches = matcher.match(des1, des2)  # 두 영상의 특징점을 매칭합니다.
matches = sorted(matches, key=lambda m: m.distance)  # 거리 기준 오름차순으로 정렬합니다.

max_draw = min(100, len(matches))  # 너무 많은 매칭을 방지하기 위해 표시 개수를 제한합니다.
match_vis_bgr = cv.drawMatches(  # 매칭 결과를 한 장의 이미지로 그립니다.
    img1_bgr,  # 첫 번째 원본 이미지를 전달합니다.
    kp1,  # 첫 번째 이미지 특징점을 전달합니다.
    img2_bgr,  # 두 번째 원본 이미지를 전달합니다.
    kp2,  # 두 번째 이미지 특징점을 전달합니다.
    matches[:max_draw],  # 상위 매칭만 시각화합니다.
    None,  # 결과 이미지를 새로 생성합니다.
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  # 매칭되지 않은 점은 생략합니다.
)  # 매칭 시각화 이미지를 생성합니다.

match_vis_rgb = cv.cvtColor(match_vis_bgr, cv.COLOR_BGR2RGB)  # matplotlib 출력을 위해 RGB로 변환합니다.

plt.figure(figsize=(16, 7))  # 출력 창 크기를 설정합니다.
plt.imshow(match_vis_rgb)  # 매칭 결과 이미지를 표시합니다.
plt.title(f"SIFT Matching ({os.path.basename(image1_path)} vs {os.path.basename(image2_path)}) - {max_draw} matches")  # 제목에 파일명과 매칭 수를 표시합니다.
plt.axis("off")  # 축을 숨깁니다.
plt.tight_layout()  # 레이아웃을 정리합니다.
plt.show()  # 화면에 결과를 출력합니다.

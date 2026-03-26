import cv2 as cv  # OpenCV를 불러옵니다.
import matplotlib.pyplot as plt  # 시각화용 matplotlib을 불러옵니다.

image_path = "mot_color70.jpg"  # 입력 이미지 파일명을 지정합니다.
image_bgr = cv.imread(image_path)  # 이미지를 BGR 형식으로 읽습니다.

if image_bgr is None:  # 이미지 로드 실패 여부를 확인합니다.
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")  # 파일 없음 예외를 발생시킵니다.

gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)  # 특징점 검출을 위해 그레이스케일로 변환합니다.
sift = cv.SIFT_create(nfeatures=1000)  # SIFT 객체를 생성하고 최대 특징점 수를 제한합니다.
keypoints, descriptors = sift.detectAndCompute(gray, None)  # 특징점과 기술자를 계산합니다.

vis_bgr = cv.drawKeypoints(  # 특징점 정보를 원본 이미지 위에 그립니다.
    image_bgr,  # 원본 BGR 이미지를 전달합니다.
    keypoints,  # 검출된 특징점 목록을 전달합니다.
    None,  # 결과 이미지를 새로 생성하도록 설정합니다.
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,  # 특징점의 크기와 방향까지 표시합니다.
)  # 특징점 시각화 결과를 생성합니다.

image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)  # 원본 이미지를 RGB로 변환합니다.
vis_rgb = cv.cvtColor(vis_bgr, cv.COLOR_BGR2RGB)  # 특징점 이미지를 RGB로 변환합니다.

plt.figure(figsize=(14, 6))  # 출력 창 크기를 설정합니다.

plt.subplot(1, 2, 1)  # 왼쪽 서브플롯을 선택합니다.
plt.imshow(image_rgb)  # 원본 이미지를 표시합니다.
plt.title("Original Image")  # 왼쪽 제목을 설정합니다.
plt.axis("off")  # 축을 숨깁니다.

plt.subplot(1, 2, 2)  # 오른쪽 서브플롯을 선택합니다.
plt.imshow(vis_rgb)  # 특징점 시각화 이미지를 표시합니다.
plt.title(f"SIFT Keypoints: {len(keypoints)}")  # 특징점 개수를 제목에 표시합니다.
plt.axis("off")  # 축을 숨깁니다.

plt.tight_layout()  # 서브플롯 간 간격을 자동 조정합니다.
plt.show()  # 화면에 결과를 출력합니다.

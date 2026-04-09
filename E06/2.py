import cv2  # 웹캠 입력과 시각화를 위해 OpenCV를 불러옵니다.
import mediapipe as mp  # 얼굴 랜드마크 검출을 위해 MediaPipe를 불러옵니다.
def main():  # 프로그램 메인 함수를 정의합니다.
    mp_face_mesh = mp.solutions.face_mesh  # FaceMesh 모듈 참조를 저장합니다.
    cap = cv2.VideoCapture(0)  # 기본 웹캠(인덱스 0)을 엽니다.
    if not cap.isOpened():  # 웹캠 열기 성공 여부를 확인합니다.
        print("웹캠을 열 수 없습니다.")  # 웹캠 실패 시 안내 메시지를 출력합니다.
        return  # 더 진행하지 않고 종료합니다.
    with mp_face_mesh.FaceMesh(  # FaceMesh 객체를 컨텍스트로 생성합니다.
        static_image_mode=False,  # 연속 영상 모드로 동작하도록 설정합니다.
        max_num_faces=1,  # 최대 검출 얼굴 수를 1개로 제한합니다.
        refine_landmarks=True,  # 눈/입술 등 세부 랜드마크 정밀도를 높입니다.
        min_detection_confidence=0.5,  # 얼굴 검출 최소 신뢰도 임계값을 설정합니다.
        min_tracking_confidence=0.5,  # 얼굴 추적 최소 신뢰도 임계값을 설정합니다.
    ) as face_mesh:  # face_mesh 이름으로 객체를 사용합니다.
        while True:  # ESC가 눌릴 때까지 프레임 처리를 반복합니다.
            ok, frame = cap.read()  # 웹캠에서 프레임을 한 장 읽습니다.
            if not ok:  # 프레임 읽기 실패 시를 확인합니다.
                break  # 루프를 종료합니다.
            frame = cv2.flip(frame, 1)  # 거울 화면처럼 좌우 반전합니다.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(rgb)  # FaceMesh 추론을 수행합니다.
            if results.multi_face_landmarks:  # 얼굴 랜드마크가 검출되었는지 확인합니다.
                h, w = frame.shape[:2]  # 현재 프레임 높이와 너비를 가져옵니다.
                for face_landmarks in results.multi_face_landmarks:  # 검출된 얼굴 랜드마크 묶음을 순회합니다.
                    for lm in face_landmarks.landmark:  # 468개 랜드마크 점을 순회합니다.
                        x = int(lm.x * w)  # 정규화 x를 픽셀 좌표로 변환합니다.
                        y = int(lm.y * h)  # 정규화 y를 픽셀 좌표로 변환합니다.
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)  # 각 랜드마크를 작은 점으로 그립니다.
            cv2.imshow("MediaPipe FaceMesh", frame)  # 결과 프레임을 화면에 출력합니다.
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 입력을 확인합니다.
                break  # ESC가 눌리면 루프를 종료합니다.
    cap.release()  # 웹캠 리소스를 해제합니다.
    cv2.destroyAllWindows()  # 열려 있는 OpenCV 창을 모두 닫습니다.
if __name__ == "__main__":  # 현재 파일이 직접 실행되는지 확인합니다.
    main()  # 메인 함수를 실행합니다.

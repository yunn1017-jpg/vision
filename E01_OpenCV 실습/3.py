import cv2 as cv # OpenCV 라이브러리를 cv라는 이름으로 가져옴
import numpy as np # 배열 처리를 위해 Numpy 라이브러리를 np라는 이름으로 가져옴

drawing = False # 마우스 드래그 상태를 추적하기 위한 불리언 변수 초기화
ix, iy = -1, -1 # 사각형 그리기를 시작할 초기 좌표를 -1로 설정
roi_selected = None # 추출된 ROI 이미지를 담을 변수를 None으로 초기화

original_img = cv.imread('soccer.jpg') # 'soccer.jpg' 파일을 읽어와서 original_img에 저장

clone = original_img.copy() # 원본 이미지 훼손 방지를 위해 동일한 복사본(clone) 생성

def select_roi(event, x, y, flags, param): # 마우스 이벤트 발생 시 호출될 콜백 함수 정의
    global drawing, ix, iy, original_img, clone, roi_selected # 외부 전역 변수들에 접근하도록 선언
    
    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼을 눌렀을 때의 이벤트 발생 시
        drawing = True # 드래그를 시작했음을 알리기 위해 drawing 변수를 True로 설정
        ix, iy = x, y # 현재 마우스 위치(x, y)를 시작 좌표(ix, iy)로 저장
        
    elif event == cv.EVENT_MOUSEMOVE: # 마우스가 움직일 때의 이벤트 발생 시
        if drawing == True: # 현재 마우스 버튼이 눌린 상태(드래그 중)라면
            original_img = clone.copy() # 잔상 제거를 위해 깨끗한 원본 복사본을 화면에 덮어씌움
            cv.rectangle(original_img, (ix, iy), (x, y), (0, 255, 0), 2) # 시작점부터 현재 마우스 위치까지 초록색 사각형을 그림
            
    elif event == cv.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼을 뗄 때의 이벤트 발생 시
        drawing = False # 드래그가 끝났음을 알리기 위해 drawing 변수를 False로 설정
        cv.rectangle(original_img, (ix, iy), (x, y), (0, 255, 0), 2) # 최종 선택된 사각형 모양을 그림
        
        start_x, end_x = min(ix, x), max(ix, x) # 사각형의 x좌표 시작점과 끝점을 올바르게 정렬
        start_y, end_y = min(iy, y), max(iy, y) # 사각형의 y좌표 시작점과 끝점을 올바르게 정렬
        
        if start_x != end_x and start_y != end_y: # 선택된 영역의 너비와 높이가 0이 아닐 경우(실제로 드래그했을 때)
            roi_selected = clone[start_y:end_y, start_x:end_x] # 복사본 이미지에서 해당 좌표 영역을 잘라내어 roi_selected에 저장
            cv.imshow('Extracted ROI', roi_selected) # 잘라낸 영역을 'Extracted ROI'라는 새 창에 표시

cv.namedWindow('Image') # 'Image'라는 이름의 창을 생성
cv.setMouseCallback('Image', select_roi) # 'Image' 창에서 발생하는 마우스 이벤트를 select_roi 함수가 처리하도록 설정

while True: # 프로그램이 종료될 때까지 무한 반복문 실행
    cv.imshow('Image', original_img) # 메인 'Image' 창에 현재 이미지를 띄움
    
    key = cv.waitKey(1) & 0xFF # 1ms 동안 키 입력을 기다리고 입력된 키 값을 저장
    
    if key == ord('q'): # 'q' 키를 눌렀을 때
        break # 무한 반복문을 빠져나감
        
    elif key == ord('r'): # 'r' 키를 눌렀을 때 (리셋)
        original_img = clone.copy() # 이미지를 원본 복사본으로 되돌려 사각형을 지움
        try: # 오류 방지를 위한 예외 처리 시작
            cv.destroyWindow('Extracted ROI') # 잘라내기 했던 창을 닫음
        except: # 오류 발생 시 (창이 이미 없을 경우)
            pass # 아무것도 하지 않고 넘어감
            
    elif key == ord('s'): # 's' 키를 눌렀을 때 (저장)
        if roi_selected is not None: # 선택된 영역이 존재한다면
            cv.imwrite('saved_roi.jpg', roi_selected) # 'saved_roi.jpg' 이름으로 잘라낸 이미지를 파일로 저장
            print("ROI 이미지가 'saved_roi.jpg'로 저장되었습니다.") # 터미널에 저장 완료 메시지 출력

cv.destroyAllWindows() # 모든 창을 닫고 프로그램을 안전하게 종료

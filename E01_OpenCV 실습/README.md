OpenCV 실습 첫 주차 프로젝트

Python과 OpenCV 라이브러리를 활용한 기초적인 이미지 처리 및 인터랙티브 도구 구현 예제를 담고 있습니다.


파일별 기능 설명

1. 1.py: 이미지 그레이스케일 변환 및 비교
이미지를 불러와 흑백(Grayscale)으로 변환한 뒤, 원본과 변환본을 한 화면에 나란히 배치하여 보여줍니다.

전체 코드:

import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # Numpy 라이브러리 임포트

img = cv.imread('soccer.jpg')# 이미지 불러오기

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)# 그레이스케일 변환

gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)# gray를 BGR로 변환 (hstack 맞추기 위해)

combined = np.hstack((img, gray_bgr))# 이미지 가로 연결

cv.imshow("Original | Grayscale", combined)# 출력

cv.waitKey(0)# 아무 키 누르면 종료
cv.destroyAllWindows() # 루프가 종료되면 모든 창을 닫고 프로그램 종료

핵심 코드:

1_이미지 색상 공간 변환 (BGR → Gray)

 gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)# 그레이스케일 변환

2_두 이미지 배열을 가로로 결합 (Stacking)

 combined = np.hstack((img, gray_bgr))# 이미지 가로 연결


2. 2.py: 마우스 드로잉 그림판 (Paint Mode)

마우스 왼쪽/오른쪽 버튼을 이용해 이미지 위에 자유롭게 선을 그릴 수 있는 간단한 그림판 프로그램입니다.

전체 코드:

import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # Numpy 라이브러리 임포트

brush_size = 5 # 붓 크기의 초기값을 5로 설정함
drawing = False # 마우스 드래그 상태
color = (255, 0, 0) # 붓의 색상 초기값 (파란색)
ix, iy = -1, -1 # 위치 저장

def paint_brush(event, x, y, flags, param): # 마우스 이벤트 처리 함수 정의
    global drawing, color, brush_size, ix, iy # 전역변수 설정
    
    if event == cv.EVENT_LBUTTONDOWN: #파란색 지정
        drawing = True # 마우스 드래그 상태
        color = (255, 0, 0) # 파란색
        ix, iy = x, y # 클릭 시작 지점 저장
        
    elif event == cv.EVENT_RBUTTONDOWN: # 빨간색 지정
        drawing = True # 마우스 드래그 상태
        color = (0, 0, 255) # 빨간색
        ix, iy = x, y # 클릭 시작 지점 저장
        
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True: # 마우스 드래그 상태
            cv.line(canvas, (ix, iy), (x, y), color, brush_size * 2) # 선 그리는 부분
            
            ix, iy = x, y # 현재 좌표를 다시 이전 좌표로 업데이트
            
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        drawing = False

canvas = cv.imread('soccer.jpg')# 캔버스(그림판) 생성


cv.namedWindow('Paint Mode') # 'Paint Mode'라는 이름의 창을 미리 생성함
cv.setMouseCallback('Paint Mode', paint_brush) # 'Paint Mode' 창에 위에서 정의한 마우스 이벤트 함수(paint_brush)를 연결함

while True: # 무한 루프 시작
    cv.imshow('Paint Mode', canvas) # 캔버스 이미지를 창에 지속적으로 업데이트하여 보여줌
    
    key = cv.waitKey(1) & 0xFF # 1밀리초 동안 키보드 입력을 대기하고 입력된 키의 ASCII 코드를 key 변수에 저장함
    
    if key == ord('q'): # 입력된 키가 소문자 'q'일 경우
        break # 무한 루프를 탈출하여 프로그램 종료 준비
        
    elif key == ord('+') or key == ord('='): # 입력된 키가 '+' 일 경우 (키보드 배열상 '='와 함께 있는 경우가 많아 예외처리 포함)
        brush_size += 1 # 붓 크기를 1 증가시킴
        if brush_size > 15: # 만약 붓 크기가 15를 초과하면
            brush_size = 15 # 붓 크기를 최대 15로 제한함
            
    elif key == ord('-'): # 입력된 키가 '-' 일 경우
        brush_size -= 1 # 붓 크기를 1 감소시킴
        if brush_size < 1: # 만약 붓 크기가 1 미만(0 이하)이 되면
            brush_size = 1 # 붓 크기를 최소 1로 제한함

cv.destroyAllWindows() # 루프가 종료되면 모든 창을 닫고 프로그램 종료

핵심 코드:

1_왼쪽 클릭: 파란색 선 그리기
    if event == cv.EVENT_LBUTTONDOWN: #파란색 지정
        drawing = True # 마우스 드래그 상태
        color = (255, 0, 0) # 파란색
        ix, iy = x, y # 클릭 시작 지점 저장

2_오른쪽 클릭: 빨간색 선 그리기
    elif event == cv.EVENT_RBUTTONDOWN: # 빨간색 지정
        drawing = True # 마우스 드래그 상태
        color = (0, 0, 255) # 빨간색
        ix, iy = x, y # 클릭 시작 지점 저장

3_키보드 조작: +/= 키로 붓 크기 증가, - 키로 감소 (최소 1 ~ 최대 15)
    elif key == ord('+') or key == ord('='): # 입력된 키가 '+' 일 경우 (키보드 배열상 '='와 함께 있는 경우가 많아 예외처리 포함)
        brush_size += 1 # 붓 크기를 1 증가시킴
        if brush_size > 15: # 만약 붓 크기가 15를 초과하면
            brush_size = 15 # 붓 크기를 최대 15로 제한함
            
    elif key == ord('-'): # 입력된 키가 '-' 일 경우
        brush_size -= 1 # 붓 크기를 1 감소시킴
        if brush_size < 1: # 만약 붓 크기가 1 미만(0 이하)이 되면
            brush_size = 1 # 붓 크기를 최소 1로 제한함

3. 3.py: 마우스 드래그를 이용한 ROI 추출

마우스로 사각형 영역을 드래그하여 관심 영역(ROI, Region of Interest)을 지정하고, 해당 부분만 별도의 이미지로 추출하거나 저장합니다.

전체 코드:

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

핵심 코드:

1_드래그 시각화: 마우스 이동에 따라 실시간으로 초록색 가이드 사각형 표시

2_영역 추출: 마우스를 떼는 순간 해당 영역을 슬라이싱하여 새 창에 표시

3_저장 및 리셋: s 키로 추출된 영역 저장, r 키로 초기화

import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # Numpy 라이브러리 임포트

# 전역 변수 초기화
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

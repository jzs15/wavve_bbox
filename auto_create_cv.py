"""
이미지 Ground Truth 생성 Program

### 사용법 ###
오른쪽 마우스 클릭 후 드래그 -> 바운딩 박스 생성
바운딩 박스 클릭 -> 바운딩 박스 선택
좌/우 방향키 -> 사진 변경 (주의!! 수정 내용 저장 안됨)
s키 -> 현재 바운딩 박스 저장
t키 -> 바둑판 모양 바운딩 박스 자동 생성
b키 -> 클래스 변경 (큰 바운딩 박스)
DEL -> 선택된 바운딩 박스 삭제
ESC -> Program 종료
"""

import os
import cv2
import numpy as np

path = './data'
img_path = os.path.join(path, 'img')
txt_path = os.path.join(path, 'txt')
ratio_list = [0.25, 1, 3]
white, blue, green, red, yellow = (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)  # 색상 값

img_list = os.listdir(img_path)
img_idx = 0
raw_img = None
img = None
config_img = np.zeros((150, 400, 3), np.uint8)

bb_list = np.empty((0, 5), np.int16)
bb_idx = -1

custom = {
    'wavve': {
        'img_width': 3840,  # 전체 이미지 너비
        'img_height': 2160,  # 전체 이미지 높이
        'bb_width': 442,  # 바운딩박스 너비
        'bb_height': 625,  # 바운딩박스 높이
        'sw': 77,  # clickable 박스 간의 x 간격
        'sh': 245,  # clickable 박스 간의 y 간격
        'bx': 36,  # 마우스 올렸을 때의 x 간격
        'by': 51  # 마우스 올렸을 때의 y 간격
    }
}
conf = custom['wavve']


# 원래 이미지 크기, 좌표 -> 변형된 좌표
def orig2trans(p):
    ratio = ratio_list[cv2.getTrackbarPos('ratio', 'config')]
    x_trans = cv2.getTrackbarPos('x_trans', 'config')
    y_trans = cv2.getTrackbarPos('y_trans', 'config')

    xy = p.copy()
    xy[::2] -= x_trans
    xy[1::2] -= y_trans
    return (xy * ratio).astype(int)


# 변형된 좌표 -> 원래 이미지 크기, 좌표
def trans2orig(p):
    ratio = ratio_list[cv2.getTrackbarPos('ratio', 'config')]
    x_trans = cv2.getTrackbarPos('x_trans', 'config')
    y_trans = cv2.getTrackbarPos('y_trans', 'config')

    xy = p.copy().astype(float)
    xy /= ratio
    xy[::2] += x_trans
    xy[1::2] += y_trans
    return xy.astype(int)


# 이미지 비율, 위치 변경
def update_image():
    global img
    ratio = ratio_list[cv2.getTrackbarPos('ratio', 'config')]
    x_trans = cv2.getTrackbarPos('x_trans', 'config')
    y_trans = cv2.getTrackbarPos('y_trans', 'config')

    num_rows, num_cols = raw_img.shape[:2]
    trans_matrix = np.float32([[1, 0, -x_trans], [0, 1, -y_trans]])
    img = cv2.warpAffine(raw_img, trans_matrix, (num_cols, num_rows), cv2.INTER_LINEAR)
    img = cv2.resize(img, None, None, ratio, ratio, cv2.INTER_CUBIC)


def get_txt():
    txt_name = img_list[img_idx][:-3] + 'txt'
    return os.path.join(txt_path, txt_name)


def update_config_image(x=0, y=0):
    updated_image = config_img.copy()
    updated_image = cv2.putText(updated_image, img_list[img_idx], (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, white, 1,
                                cv2.LINE_AA)
    updated_image = cv2.putText(updated_image, 'x: {}'.format(x), (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, white, 1,
                                cv2.LINE_AA)
    updated_image = cv2.putText(updated_image, 'y: {}'.format(y), (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, white, 1,
                                cv2.LINE_AA)
    cv2.imshow('config', updated_image)


# 이미지 불러오기
def change_image():
    global bb_list, raw_img
    clear_bb()

    print(img_list[img_idx])
    update_config_image()

    raw_img = cv2.imread(os.path.join(img_path, img_list[img_idx]))
    update_image()

    txt = get_txt()
    if os.path.isfile(txt):
        f = open(txt, 'r')
        while True:
            line = f.readline().strip()
            if not line:
                break
            c, x1, y1, x2, y2 = map(int, line.split(' '))
            bb_list = np.append(bb_list, [[c, x1, y1, x2, y2]], axis=0)


def clear_bb():
    global bb_list, bb_idx
    bb_list = np.empty((0, 5), np.int16)
    bb_idx = -1


# 바운딩 박스 선택
def select_bb(x, y):
    x, y = trans2orig(np.array([x, y]))
    for i, (_, x1, y1, x2, y2) in enumerate(bb_list):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return -1


# 이미지 그리기
def draw_img(saved=False):
    overlay = img.copy()  # 사각형 그림 표현을 위한 이미지 복제
    img_draw = img.copy()
    for i, v in enumerate(bb_list):
        c = v[0]
        x1, y1, x2, y2 = orig2trans(v[1:])

        if c == 0:
            if saved:
                color = green
            elif i == bb_idx:
                color = red
            else:
                color = blue
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        else:
            if saved:
                color = green
            elif i == bb_idx:
                color = red
            else:
                color = yellow
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    alpha = 0.3
    img_draw = cv2.addWeighted(overlay, alpha, img_draw, 1 - alpha, 0)
    cv2.imshow('img', img_draw)  # 사각형 표시된 그림 화면 출력
    return img_draw


def save_bb():
    txt = get_txt()
    f = open(txt, 'w+')
    for i, (c, x1, y1, x2, y2) in enumerate(bb_list):
        f.write('{} {} {} {} {}\n'.format(c, x1, y1, x2, y2))
    f.close()


# 바둑판 모양 바운딩박스 생성
def auto_bb():
    global bb_idx, bb_list
    if bb_idx < 0 or bb_idx >= len(bb_list):
        return
    new_bb_list = []
    c, x1, y1, x2, y2 = bb_list[bb_idx]

    if c != 0:  # 기본 바운딩 박스가 아니면 실행하지 않음
        return

    w, h = x2 - x1, y2 - y1

    sw, sh = w + conf['sw'], h + conf['sh']
    xs, ys = x1 % sw, y1 % sh

    if xs > 0:
        xs -= sw
    if ys > 0:
        ys -= sh

    img_width, img_height = conf['img_width'], conf['img_height']
    for y in range(ys, img_height, sh):
        for x in range(xs, img_width, sw):
            new_bb_list.append([c, max(0, x), max(0, y), min(img_width - 1, x + w), min(img_height - 1, y + h)])

    bb_list = np.delete(bb_list, bb_idx, axis=0)
    bb_list = np.append(bb_list, new_bb_list, axis=0).astype(int)
    bb_idx = len(bb_list) - 1


# 포커스된 파운딩 박스로 변경
def on_bb():
    global bb_idx, bb_list
    if bb_idx < 0 or bb_idx >= len(bb_list):
        return
    c, x1, y1, x2, y2 = bb_list[bb_idx]

    if c != 0:
        return

    x1 = max(0, x1 - conf['bx'])
    x2 = min(conf['img_width'] - 1, x2 + conf['bx'])
    y1 = max(0, y1 - conf['by'])
    y2 = min(conf['img_height'] - 1, y2 + conf['by'])

    bb_list = np.delete(bb_list, bb_idx, axis=0)
    bb_list = np.append(bb_list, [[0, x1, y1, x2, y2], [1, x1, y1, x2, y2]], axis=0).astype(int)
    bb_idx = -1


def on_mouse(event, x, y, flags, param):  # 마우스 이벤트 핸들 함수
    global bb_idx, bb_list  # 전역변수 참조
    if event == cv2.EVENT_RBUTTONDOWN:
        drawable = cv2.getTrackbarPos('Drawable', 'config')
        cv2.setTrackbarPos('Drawable', 'config', abs(drawable - 1))
    elif event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 다운, 바운딩 박스 생성 or 선택
        drawable = cv2.getTrackbarPos('Drawable', 'config')
        if drawable == 1:
            x1, y1 = trans2orig(np.array([x, y]))
            bb_list = np.append(bb_list, [
                [0, max(0, x1), max(0, y1), min(conf['img_width'], x1 + conf['bb_width']),
                 min(conf['img_height'], y1 + conf['bb_height'])]], axis=0).astype(int)
            bb_idx = len(bb_list) - 1
            draw_img()
        else:
            bb_idx = select_bb(x, y)
            draw_img()
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 움직임 ---③
        drawable = cv2.getTrackbarPos('Drawable', 'config')
        if drawable == 1:
            print('X: {}, Y: {}'.format(x, y))
            update_config_image(x, y)


def nothing(x):
    pass


def track_draw(x):
    update_image()
    draw_img()


def change_image_idx(x):
    global img_idx
    img_idx = x
    change_image()
    draw_img()


cv2.namedWindow('config')
cv2.namedWindow('img')
cv2.setMouseCallback('img', on_mouse)
cv2.createTrackbar('Drawable', 'config', 0, 1, nothing)
cv2.createTrackbar('ratio', 'config', 0, len(ratio_list) - 1, track_draw)
cv2.createTrackbar('x_trans', 'config', 0, conf['img_width'], track_draw)
cv2.createTrackbar('y_trans', 'config', 0, conf['img_height'], track_draw)
cv2.createTrackbar('img_idx', 'config', 0, len(img_list) - 1, change_image_idx)

change_image()
draw_img()

while True:  # 무한 루프
    keycode = cv2.waitKeyEx()  # 키보드 입력 반환 값 저장

    if keycode == 0x2e0000:  # Delete
        if 0 <= bb_idx < len(bb_list):
            bb_list = np.delete(bb_list, bb_idx, axis=0)
            bb_idx = -1
            draw_img()
    elif keycode == 0x250000:  # Left
        cv2.setTrackbarPos('img_idx', 'config', (img_idx - 1) % len(img_list))
    elif keycode == 0x270000:  # Right
        cv2.setTrackbarPos('img_idx', 'config', (img_idx + 1) % len(img_list))
    elif keycode == ord('x') or keycode == ord('x'):
        clear_bb()
        draw_img()
    elif keycode == ord('s') or keycode == ord('S'):
        bb_idx = -1
        save_bb()
        draw_img(saved=True)
    elif keycode == ord('b') or keycode == ord('B'):
        on_bb()
        draw_img()
    elif keycode == ord('t') or keycode == ord('T'):
        auto_bb()
        draw_img()
    elif keycode == ord('r') or keycode == ord('R'):
        ratio = cv2.getTrackbarPos('ratio', 'config')
        if ratio != 0:
            cv2.setTrackbarPos('ratio', 'config', 0)
            cv2.setTrackbarPos('x_trans', 'config', 0)
            cv2.setTrackbarPos('y_trans', 'config', 0)
        else:
            cv2.setTrackbarPos('ratio', 'config', len(ratio_list) - 1)
            cv2.setTrackbarPos('x_trans', 'config', 0)
            cv2.setTrackbarPos('y_trans', 'config', 550)
        draw_img()
    elif keycode == 0x1B:  # ESC 누를 시 종료
        break

cv2.destroyAllWindows()

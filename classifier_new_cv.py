"""
이미지 Ground Truth 생성 Program

### 사용법 ###
1. data/class 폴더에 클래스를 대표하는 이미지 추가 (파일 이름은 {세자리 숫자}.jpg)
2. class_num을 클래스 번호로 설정
3. 아래 키 혹은 config 창을 활용하여 바운딩 박스 생성
4. s키를 눌러 저장

저장 위치: data/txt
저장 파일 명: {Object 수}_{클래스 번호}_{Focus 된 오브젝트}.txt
Focus 된 오브젝트는 0부터 Object 수까지 자동 생성

오른쪽 마우스 클릭 -> 선택 모드, 그리기 모드 변경
왼쪽 마우스 클릭 -> 바운딩 박스 선택 or 그리기
방향키 -> 해당 방향으로 바운딩 박스 생성
r키 -> 미리 설정된 화면 비율 변경 및 위치 이동 / 초기화
s키 -> 현재 바운딩 박스 저장
x키 -> 모든 바운딩 박스 지우기
b키 -> 선택된 바운딩 박스 focus 된 모양 확인
DEL -> 선택된 바운딩 박스 삭제
ESC -> Program 종료

### 다른 APP 대응 ###
img_width
img_height
bbox_type
세 가지 변수 변경 후 사용
"""

import os
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

class_num = 100
img_name = '{:03d}.jpg'.format(class_num)
path = './data'
img_path = os.path.join(path, 'class')
ratio_list = [0.25, 1, 3]
white, blue, green, red, yellow = (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)  # 색상 값

raw_img = None
img = None
config_img = np.zeros((150, 400, 3), np.uint8)

bb_list = np.empty((0, 5), np.int16)
bb_idx = -1
draw_focused = False

img_width = 3840
img_height = 2160

bbox_type = [
    {
        'name': '세로로 긴 이미지',
        'bb_width': 442,  # 바운딩박스 너비
        'bb_height': 625,  # 바운딩박스 높이
        'sw': 77,  # clickable 박스 간의 x 간격
        'sh': 245,  # clickable 박스 간의 y 간격
        'bx': 36,  # 마우스 올렸을 때의 x 간격
        'by': 51  # 마우스 올렸을 때의 y 간격
    },
    {
        'name': '가로로 긴 이미지, 카테고리',
        'bb_width': 700,  # 바운딩박스 너비
        'bb_height': 400,  # 바운딩박스 높이
        'sw': 80,  # clickable 박스 간의 x 간격
        'sh': 220,  # clickable 박스 간의 y 간격
        'bx': 57,  # 마우스 올렸을 때의 x 간격
        'by': 33  # 마우스 올렸을 때의 y 간격
    },
    {
        'name': '카드 형태',
        'bb_width': 700,  # 바운딩박스 너비
        'bb_height': 600,  # 바운딩박스 높이
        'sw': 80,  # clickable 박스 간의 x 간격
        'sh': 245,  # clickable 박스 간의 y 간격
        'bx': 57,  # 마우스 올렸을 때의 x 간격
        'by': 31  # 마우스 올렸을 때의 y 간격
    },
    {
        'name': '재생 목록',
        'bb_width': 450,  # 바운딩박스 너비
        'bb_height': 250,  # 바운딩박스 높이
        'sw': 99999,  # clickable 박스 간의 x 간격 (좌 우 생성되지 않도록 함)
        'sh': 81,  # clickable 박스 간의 y 간격
        'bx': 0,  # 마우스 올렸을 때의 x 간격
        'by': 42  # 마우스 올렸을 때의 y 간격
    },
    {
        'name': '세로 이미지 (카테고리)',
        'bb_width': 442,  # 바운딩박스 너비
        'bb_height': 625,  # 바운딩박스 높이
        'sw': 77,  # clickable 박스 간의 x 간격
        'sh': 80,  # clickable 박스 간의 y 간격
        'bx': 36,  # 마우스 올렸을 때의 x 간격
        'by': 51  # 마우스 올렸을 때의 y 간격
    },
    {
        'name': '카드 형태 (카테고리)',
        'bb_width': 700,  # 바운딩박스 너비
        'bb_height': 600,  # 바운딩박스 높이
        'sw': 80,  # clickable 박스 간의 x 간격
        'sh': 100,  # clickable 박스 간의 y 간격
        'bx': 57,  # 마우스 올렸을 때의 x 간격
        'by': 31  # 마우스 올렸을 때의 y 간격
    },
]


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


def update_config_image(x=0, y=0):
    updated_image = config_img.copy()
    updated_image = cv2.putText(updated_image, img_name, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, white, 1,
                                cv2.LINE_AA)
    updated_image = cv2.putText(updated_image, 'Object: {}'.format(len(bb_list)), (200, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                                white, 1, cv2.LINE_AA)
    updated_image = cv2.putText(updated_image, 'x: {}'.format(x), (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, white, 1,
                                cv2.LINE_AA)
    updated_image = cv2.putText(updated_image, 'y: {}'.format(y), (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, white, 1,
                                cv2.LINE_AA)

    font_path = "fonts/gulim.ttc"
    font = ImageFont.truetype(font_path, 30)
    img_pil = Image.fromarray(updated_image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 100), 'BBOX: {}'.format(bbox_type[cv2.getTrackbarPos('bbox_type', 'config')]['name']), font=font,
              fill=white)
    updated_image = np.array(img_pil)
    # updated_image = cv2.putText(updated_image, draw, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, white, 1, cv2.LINE_AA)
    cv2.imshow('config', updated_image)


# 이미지 불러오기
def change_image():
    global bb_list, raw_img
    clear_bb()
    update_config_image()
    raw_img = cv2.imread(os.path.join(img_path, img_name))
    update_image()


def clear_bb():
    global bb_list, bb_idx
    bb_list = np.empty((0, 5), np.int16)
    bb_idx = -1


def get_focused_bbox(v):
    c = v[0]
    x1, y1, x2, y2 = v[1:]
    x1 = max(0, x1 - bbox_type[c]['bx'])
    x2 = min(img_width - 1, x2 + bbox_type[c]['bx'])
    if c == 3:  # 재생 목록
        x1 = max(0, x1 - 160)
        x2 = min(img_width - 1, x2 + 830)
    y1 = max(0, y1 - bbox_type[c]['by'])
    y2 = min(img_height - 1, y2 + bbox_type[c]['by'])
    return np.array([x1, y1, x2, y2])


# 이미지 그리기
def draw_img(saved=False):
    img_draw = img.copy()  # 사각형 그림 표현을 위한 이미지 복제
    overlay = img.copy()
    alpha = 0.3
    for i, v in enumerate(bb_list):
        x1, y1, x2, y2 = orig2trans(v[1:])
        thickness = 2
        draw_color = blue
        if saved:
            draw_color = green
        elif i == bb_idx:
            draw_color = red
            if draw_focused:
                x1, y1, x2, y2 = orig2trans(get_focused_bbox(v))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), yellow, -1)
                continue

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), draw_color, thickness)

    img_draw = cv2.addWeighted(overlay, alpha, img_draw, 1 - alpha, 0)
    cv2.imshow('img', img_draw)  # 사각형 표시된 그림 화면 출력
    return img_draw


# 바운딩 박스 선택
def select_bb(x, y):
    x, y = trans2orig(np.array([x, y]))
    for i, (_, x1, y1, x2, y2) in enumerate(bb_list):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return -1


def save_bb():
    res = sorted(bb_list, key=lambda x: (x[2], x[1], x[0]))
    object_cnt = len(res)
    for fi in range(object_cnt + 1):
        txt_name = '{:03d}_{:03d}_{:03d}.txt'.format(object_cnt, class_num, fi)
        txt = os.path.join('./data/txt', txt_name)
        f = open(txt, 'w+')
        for bi, v in enumerate(res):
            c, x1, y1, x2, y2 = v
            if bi == fi - 1:
                x1, y1, x2, y2 = get_focused_bbox(v)
                f.write(
                    '{} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(1, (x1 + x2) / 2 / img_width,
                                                              (y1 + y2) / 2 / img_height,
                                                              (x2 - x1) / img_width, (y2 - y1) / img_height))
            f.write(
                '{} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(0, (x1 + x2) / 2 / img_width,
                                                          (y1 + y2) / 2 / img_height,
                                                          (x2 - x1) / img_width, (y2 - y1) / img_height))

        f.close()


# 바둑판 모양 바운딩박스 생성
def auto_bb(dir):
    global bb_idx, bb_list
    if bb_idx < 0 or bb_idx >= len(bb_list):
        return
    new_bb_list = []
    c, x1, y1, x2, y2 = bb_list[bb_idx]

    w = bbox_type[c]['bb_width']
    h = bbox_type[c]['bb_height']
    sw = bbox_type[c]['sw']
    sh = bbox_type[c]['sh']

    if dir == 0:
        gap = w + sw
        xs = x1 % gap
        if xs > 0:
            xs -= gap
        for x in range(xs, x1, gap):
            # new_bb_list.append([c, max(0, x), max(0, y1), min(img_width - 1, x + w), min(img_height - 1, y1 + h)])
            new_bb_list.append([c, max(0, x), max(0, y1), min(img_width - 1, x + w), min(img_height - 1, y2)])
    elif dir == 1:
        gap = w + sw
        xs = x1 + gap
        for x in range(xs, img_width, gap):
            new_bb_list.append([c, max(0, x), max(0, y1), min(img_width - 1, x + w), min(img_height - 1, y2)])
    elif dir == 2:
        gap = h + sh
        ys = y1 % gap
        if ys > 0:
            ys -= gap
        for y in range(ys, y1, gap):
            new_bb_list.append([c, max(0, x1), max(0, y), min(img_width - 1, x2), min(img_height - 1, y + h)])
    elif dir == 3:
        gap = h + sh
        ys = y1 + gap
        for y in range(ys, img_height, gap):
            new_bb_list.append([c, max(0, x1), max(0, y), min(img_width - 1, x2), min(img_height - 1, y + h)])

    bb_list = np.append(bb_list, new_bb_list, axis=0).astype(int)
    draw_img()
    update_config_image()


def on_mouse(event, x, y, flags, param):  # 마우스 이벤트 핸들 함수
    global bb_idx, bb_list  # 전역변수 참조
    if event == cv2.EVENT_RBUTTONDOWN:
        drawable = cv2.getTrackbarPos('Drawable', 'config')
        cv2.setTrackbarPos('Drawable', 'config', abs(drawable - 1))
    elif event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 다운, 바운딩 박스 생성 or 선택
        drawable = cv2.getTrackbarPos('Drawable', 'config')
        if drawable == 1:
            c = cv2.getTrackbarPos('bbox_type', 'config')
            x1, y1 = trans2orig(np.array([x, y]))
            bb_list = np.append(bb_list, [
                [c, max(0, x1), max(0, y1), min(img_width, x1 + bbox_type[c]['bb_width']),
                 min(img_height, y1 + bbox_type[c]['bb_height'])]], axis=0).astype(int)
            bb_idx = len(bb_list) - 1
            draw_img()
            update_config_image(x, y)
            cv2.setTrackbarPos('Drawable', 'config', 0)
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


cv2.namedWindow('config')
cv2.namedWindow('img')
cv2.setMouseCallback('img', on_mouse)
cv2.createTrackbar('Drawable', 'config', 0, 1, nothing)
cv2.createTrackbar('bbox_type', 'config', 0, len(bbox_type) - 1, update_config_image)
cv2.createTrackbar('ratio', 'config', 0, len(ratio_list) - 1, track_draw)
cv2.createTrackbar('x_trans', 'config', 0, img_width, track_draw)
cv2.createTrackbar('y_trans', 'config', 0, img_height, track_draw)

change_image()
draw_img()

while True:  # 무한 루프
    keycode = cv2.waitKeyEx()  # 키보드 입력 반환 값 저장

    if keycode == 0x2e0000:  # Delete
        if 0 <= bb_idx < len(bb_list):
            bb_list = np.delete(bb_list, bb_idx, axis=0)
            bb_idx = -1
            draw_img()
            update_config_image()
    elif keycode == ord('q') or keycode == ord('Q'):  # 이전 이미지
        # cv2.setTrackbarPos('img_idx', 'config', (img_idx - 1) % len(img_list))
        pass
    elif keycode == ord('w') or keycode == ord('W'):  # 다음 이미지
        # cv2.setTrackbarPos('img_idx', 'config', (img_idx + 1) % len(img_list))
        pass
    elif keycode == 0x250000:  # Left
        auto_bb(0)
    elif keycode == 0x270000:  # Right
        auto_bb(1)
    elif keycode == 0x260000:  # Up
        auto_bb(2)
    elif keycode == 0x280000:  # Down
        auto_bb(3)
    elif keycode == ord('x') or keycode == ord('x'):
        clear_bb()
        draw_img()
    elif keycode == ord('s') or keycode == ord('S'):
        bb_idx = -1
        save_bb()
        draw_img(saved=True)
    elif keycode == ord('b') or keycode == ord('B'):
        draw_focused = not draw_focused
        draw_img()
    elif keycode == ord('r') or keycode == ord('R'):
        if cv2.getTrackbarPos('ratio', 'config') != 0:
            cv2.setTrackbarPos('ratio', 'config', 0)
            cv2.setTrackbarPos('x_trans', 'config', 0)
            cv2.setTrackbarPos('y_trans', 'config', 0)
        else:
            cv2.setTrackbarPos('ratio', 'config', len(ratio_list) - 1)
            cv2.setTrackbarPos('x_trans', 'config', 250)
            cv2.setTrackbarPos('y_trans', 'config', 1550)
            cv2.setTrackbarPos('bbox_type', 'config', 1)
            # cv2.setTrackbarPos('y_trans', 'config', 550)
        draw_img()
    elif keycode == 0x1B:  # ESC 누를 시 종료
        break

cv2.destroyAllWindows()

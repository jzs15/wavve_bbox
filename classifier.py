import os.path
import sys
import cv2
import json
import glob
import shutil
import natsort
import numpy as np
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtCore import Qt, QRectF

img_width = 3840
img_height = 2160


class ClassifierApp(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = uic.loadUi("./classifier.ui", self)

        self.cur_scene = QtWidgets.QGraphicsScene(self.ui.cur_image_view)
        self.ui.cur_image_view.setScene(self.cur_scene)

        self.cur_image_list = natsort.natsorted(os.listdir('./data/img'))
        self.cur_image_idx = 0
        self.cur_image = None
        self.txt_path = None
        self.bbox_list = np.empty((0, 5), np.int16)

        self.image_dict = {}
        self.init_image_dict()

        self.ui.show()
        self.show_image()

    def init_image_dict(self):
        if not os.path.isfile('./data/image_dict.json'):
            f = open('./data/image_dict.json', 'w')
            json.dump({}, f)
            f.close()
        with open('./data/image_dict.json', 'r') as file:
            self.image_dict = json.load(file)

    def write_image_dict(self):
        with open('./data/image_dict.json', 'w', encoding='utf-8') as file:
            json.dump(self.image_dict, file)

    def save_bbox(self, txt_path):
        f = open(txt_path, 'w+')
        for c, x1, y1, x2, y2 in self.bbox_list:
            f.write(
                '{} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(int(c), (x1 + x2) / 2 / img_width, (y1 + y2) / 2 / img_height,
                                                          (x2 - x1) / img_width, (y2 - y1) / img_height))
        f.close()

    def add_image(self):
        cur_name = self.cur_image_list[self.cur_image_idx]

        image_list = glob.glob(
            './data/res/type_{0:03d}_{1:03d}/images/{0:03d}_{1:03d}_{2:03d}_*'.format(self.ui.object_num.value(),
                                                                                      self.ui.image_class.value(),
                                                                                      self.ui.focused_object.value()))
        if len(image_list) > 0:
            last_image = os.path.splitext(os.path.basename(image_list[-1]))[0]
            new_id = int(last_image.split('_')[-1]) + 1
        else:
            new_id = 0

        type_name = 'type_{:03d}_{:03d}'.format(self.ui.object_num.value(), self.ui.image_class.value())
        new_name = '{:03d}_{:03d}_{:03d}_{:03d}'.format(self.ui.object_num.value(), self.ui.image_class.value(),
                                                        self.ui.focused_object.value(), new_id)

        # 타입 폴더 없으면 생성
        new_path = os.path.join('./data/res', type_name)
        if not os.path.isdir(new_path):
            os.makedirs(os.path.join(new_path, 'images'))
            os.makedirs(os.path.join(new_path, 'labels'))

        # 이미지 저장
        shutil.copyfile(os.path.join('./data/img', cur_name), os.path.join(new_path, 'images', new_name + '.jpg'))

        # label 저장
        if self.ui.extra_check.isChecked():
            # extra_check: 더보기가 선택된 이미지(새로운 BBox 생성)
            self.save_bbox(os.path.join(new_path, 'labels', new_name + '.txt'))
        else:
            shutil.copyfile(os.path.join('./data/txt', new_name[:-4] + '.txt'),
                            os.path.join(new_path, 'labels', new_name + '.txt'))
        self.image_dict[cur_name] = new_name
        self.ui.cur_saved_name.setText(new_name + '.jpg')
        self.write_image_dict()
        self.update_image()

    def update_bbox(self):
        self.bbox_list = np.empty((0, 5), np.int16)
        is_view = self.ui.bbox_check.isChecked()
        if not is_view or self.txt_path is None:
            return

        f = open(self.txt_path, 'r')
        while True:
            line = f.readline().strip()
            if not line:
                break
            c, x, y, w, h = map(float, line.split(' '))
            c = int(c)
            x1 = int((x - w / 2) * img_width)
            y1 = int((y - h / 2) * img_height)
            x2 = int((x + w / 2) * img_width)
            y2 = int((y + h / 2) * img_height)
            self.bbox_list = np.append(self.bbox_list, [[c, x1, y1, x2, y2]], axis=0)
        f.close()

        if self.ui.extra_check.isChecked():
            fv = [-1, -1, -1, -1]
            fy = -1
            for c, x1, y1, x2, y2 in self.bbox_list:
                if c == 1:
                    fv = [x1, y1, x2, y2]
                    fy = min(img_width - 1, y2 + 29)
                    break

            for i, v in enumerate(self.bbox_list):
                if (v[1:] == fv).all():
                    self.bbox_list[i][4] = fy

    def search(self):
        txt_name = '{:03d}_{:03d}_{:03d}.txt'.format(self.ui.object_num.value(), self.ui.image_class.value(),
                                                     self.ui.focused_object.value())
        txt_path = os.path.join('./data/txt/', txt_name)
        if os.path.exists(txt_path):
            self.ui.anno_status.setText('있음')
            self.ui.anno_status.setStyleSheet("color: black;")
            self.txt_path = txt_path
            self.ui.add_button.setEnabled(True)
        else:
            self.ui.anno_status.setText('없음')
            self.ui.anno_status.setStyleSheet("color: red;")
            self.txt_path = None
            self.ui.add_button.setEnabled(False)
        self.show_image()

    def next_image(self):
        only_not = self.ui.only_not_added.isChecked()
        if only_not:
            for idx in range(self.cur_image_idx + 1, len(self.cur_image_list)):
                if self.cur_image_list[idx] not in self.image_dict:
                    self.cur_image_idx = idx
                    break
        elif self.cur_image_idx < len(self.cur_image_list) - 1:
            self.cur_image_idx += 1
        self.update_image()
        self.show_image()

    def prev_image(self):
        only_not = self.ui.only_not_added.isChecked()
        if only_not:
            for idx in range(self.cur_image_idx - 1, -1, -1):
                if self.cur_image_list[idx] not in self.image_dict:
                    self.cur_image_idx = idx
                    break
        elif self.cur_image_idx > 0:
            self.cur_image_idx -= 1
        self.update_image()
        self.show_image()

    def update_image(self):
        cur_name = self.cur_image_list[self.cur_image_idx]
        self.cur_image = cv2.imread(os.path.join('./data/img', cur_name))
        self.ui.cur_image_name.setText(cur_name)
        if cur_name in self.image_dict:
            self.ui.cur_saved_name.setText(self.image_dict[cur_name] + '.jpg')
            self.ui.add_button.setEnabled(False)
        else:
            self.ui.cur_saved_name.setText('-')

    def show_image(self):
        self.cur_scene.clear()
        if self.cur_image is None:
            self.update_image()

        image_draw = self.cur_image.copy()
        overlay = self.cur_image.copy()
        self.update_bbox()
        for c, x1, y1, x2, y2 in self.bbox_list:
            if c == 1:
                draw_color = (0, 255, 255)
                thickness = -1
                cv2.rectangle(overlay, (x1, y1), (x2, y2), draw_color, thickness)
            else:
                draw_color = (0, 255, 0)
                thickness = 8
                cv2.rectangle(image_draw, (x1, y1), (x2, y2), draw_color, thickness)

        image_draw = cv2.addWeighted(overlay, 0.3, image_draw, 1 - 0.3, 0)
        image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
        h, w, c = image_draw.shape
        qt_img = QtGui.QImage(image_draw.data, w, h, w * c, QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(qt_img)

        self.cur_scene.addPixmap(pixmap)
        self.ui.cur_image_view.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)
        self.cur_scene.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    classifier = ClassifierApp()
    sys.exit(app.exec())

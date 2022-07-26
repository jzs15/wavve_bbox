import os.path
import sys
import cv2
import json
import glob
import shutil
import natsort
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtCore import Qt, QRectF


class ViewerApp(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = uic.loadUi("./viewer.ui", self)

        self.cur_scene = QtWidgets.QGraphicsScene(self.ui.cur_image_view)
        self.ui.cur_image_view.setScene(self.cur_scene)

        self.cur_image_list = glob.glob('data/res/type*/images/*.jpg')
        self.ui.image_num.setText(str(len(self.cur_image_list)))
        self.ui.cur_idx.setMaximum(len(self.cur_image_list))
        self.cur_image_idx = 0
        self.cur_image = None
        self.bbox_list = np.empty((0, 5), np.int16)

        self.ui.show()
        self.show_image()

    def keyPressEvent(self, a0: QtGui.QKeyEvent):
        if a0.key() == Qt.Key_Q:
            self.prev_image()
        elif a0.key() == Qt.Key_W:
            self.next_image()

    def search(self, idx):
        if idx > len(self.cur_image_list):
            idx = len(self.cur_image_list)
        elif idx <= 0:
            idx = 1
        self.cur_image_idx = idx - 1
        self.update_image()
        self.show_image()

    def update_bbox(self):
        self.bbox_list = np.empty((0, 5), np.int16)
        is_view = self.ui.bbox_check.isChecked()
        if not is_view:
            return
        cur_name = self.cur_image_list[self.cur_image_idx][-19:-4]
        txt_path = './data/res/type_{}/labels/{}.txt'.format(cur_name[:7], cur_name)

        img_width = 3840
        img_height = 2160

        f = open(txt_path, 'r')
        while True:
            line = f.readline().strip()
            if not line:
                break
            c, x, y, w, h = map(float, line.split(' '))
            self.bbox_list = np.append(self.bbox_list, [
                [int(c), int((x - w / 2) * img_width), int((y - h / 2) * img_height), int((x + w / 2) * img_width),
                 int((y + h / 2) * img_height)]], axis=0)
        f.close()

    def next_image(self):
        if self.cur_image_idx < len(self.cur_image_list) - 1:
            self.cur_image_idx += 1
        self.update_image()
        self.show_image()

    def prev_image(self):
        if self.cur_image_idx > 0:
            self.cur_image_idx -= 1
        self.update_image()
        self.show_image()

    def update_image(self):
        self.ui.cur_idx.setValue(self.cur_image_idx + 1)
        cur_name = self.cur_image_list[self.cur_image_idx][-19:]
        self.cur_image = cv2.imread(self.cur_image_list[self.cur_image_idx])
        self.ui.cur_image_name.setText(cur_name)

    def show_image(self):
        self.cur_scene.clear()
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
    classifier = ViewerApp()
    sys.exit(app.exec())

import sys
from PyQt5 import QtWidgets, uic


class ClassifierApp(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = uic.loadUi("./classifier.ui", self)
        self.ui.showMaximized()

    def add_image(self):
        pass

    def modify_image(self):
        pass

    def view_bbox(self, is_view):
        pass

    def search(self):
        pass

    def next_class(self):
        pass

    def prev_class(self):
        pass

    def next_image(self):
        pass

    def prev_image(self):
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    classifier = ClassifierApp()
    sys.exit(app.exec())

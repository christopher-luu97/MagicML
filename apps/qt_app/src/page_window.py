from PyQt6.QtWidgets import (QMainWindow)
from PyQt6.QtCore import *
class PageWindow(QMainWindow):
    gotoSignal = pyqtSignal(str)

    def goto(self, name):
        self.gotoSignal.emit(name)
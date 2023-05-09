# Entry point to the application

import os

from PyQt6.QtWidgets import (QMainWindow, QApplication, QStackedWidget)
from PyQt6.QtCore import *

from apps.qt_app.mainWIndow import MainWindow
from apps.qt_app.splashWindow import splashWindow
from apps.qt_app.signals import PageWindow

class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.m_pages = {}

        mw = MainWindow()

        self.register(mw, "main")
        self.goto("main")

    def register(self, widget, name):
        self.m_pages[name] = widget
        self.stacked_widget.addWidget(widget)
        if isinstance(widget, PageWindow):
            widget.gotoSignal.connect(self.goto)

    @pyqtSlot(str)
    def goto(self, name):
        if name in self.m_pages:
            widget = self.m_pages[name]
            self.stacked_widget.setCurrentWidget(widget)
            self.setWindowTitle(widget.windowTitle())

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    basedir = os.path.dirname(sys.argv[0])
    style_dir = os.path.join(basedir, 'styles', 'style.qss') # dev

    with open(style_dir, 'r') as f:
        style = f.read()
    app.setStyleSheet(style)
    sp = splashWindow()
    sp.show()
    sp.progress()
    w = Window()
    w.showMaximized()
    w.show()
    sp.finish(w)
    sys.exit(app.exec())
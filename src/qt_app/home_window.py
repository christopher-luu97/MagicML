from PyQt6.QtWidgets import QWidget
import os 
from PyQt6 import uic
import sys


class HomeWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        basedir = os.path.dirname(sys.argv[0]) # os.getcwd()
        ui_dir = os.path.join(basedir, 'widgets', 'home_window.ui')
        
        ui = uic.loadUi(ui_dir, self)
        self.ui = ui

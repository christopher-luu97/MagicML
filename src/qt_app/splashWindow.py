import time
import os
from PyQt6.QtWidgets import QSplashScreen, QProgressBar
from PyQt6 import uic
from PyQt6.QtCore import *

class splashWindow(QSplashScreen):
    def __init__(self):
        super(splashWindow, self).__init__()

        basedir = os.getcwd() # os.path.dirname(sys.argv[0])
        ui_dir = os.path.join(basedir, 'ui', 'splash.ui')
        
        ui = uic.loadUi(ui_dir, self)
        self.progressBar = self.findChild(QProgressBar, 'progressBar')
        # Remove windows default title bar
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    
    def progress(self):
        """
        Progress bar to increment
        """
        i = 1
        while i <= 100:
            time.sleep(0.1)
            self.progressBar.setValue(i)
            i +=1

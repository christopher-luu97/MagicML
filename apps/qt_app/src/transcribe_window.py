from src.page_window import PageWindow
from PyQt6.QtGui import QAction
import os
import sys
from PyQt6 import uic

class TranscribeWindow(PageWindow):
    def __init__(self):
        super(TranscribeWindow, self).__init__()
        basedir = os.path.dirname(sys.argv[0]) # os.getcwd()
        ui_dir = os.path.join(basedir, 'widgets', 'transcribe.ui')
        print(ui_dir)
        
        ui = uic.loadUi(ui_dir, self)
        self.ui = ui
        self.chatgpt_window = self.findChild(QAction, 'actionChatGPT')
        self.chatgpt_window.triggered.connect(self.navigate_to_chatgpt)

        self.transcribe_window = self.findChild(QAction, 'actionTranscribe')
        self.transcribe_window.triggered.connect(self.navigate_to_transcribe)

    def navigate_to_chatgpt(self):
        """_summary_
        """
        self.goto("main")
    
    def navigate_to_transcribe(self):
        """
        """
        self.goto("transcribe")
    
    def goToMain(self):
        self.goto("main")
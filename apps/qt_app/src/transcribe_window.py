from src.page_window import PageWindow

class TranscribeWindow(PageWindow):
    def __init__(self):
        super(TranscribeWindow, self).__init__()
    
    def goToMain(self):
        self.goto("main")
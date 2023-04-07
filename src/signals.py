from PyQt6.QtCore import *
from PyQt6.QtWidgets import QMainWindow

class dialogCommunicate(QObject):
    """
    Capture signals from dialogs.
    To be used to communicate across dialogs

    Args:
        QObject (_type_): _description_
    """
    closure = pyqtSignal(int)

class PageWindow(QMainWindow):
    """

    Args:
        QMainWindow (_type_): _description_
    """
    gotoSignal = pyqtSignal(str)

    def goto(self, name):
        self.gotoSignal.emit(name)
from PyQt6.QtWidgets import (QDialog, QLabel, QDialogButtonBox, 
                             QTextBrowser, QPushButton, QLineEdit,
                             QMessageBox)
from PyQt6 import uic
from PyQt6.QtCore import *
import os
import sys
import subprocess as sp

from authoring.author import Author
from authoring.author_metadata import magic_metadata
from signals import dialogCommunicate

class generalDialog(QDialog):
    """
    General dialog class that can be called to start the other dialogs.
    Instead of having multiple classes, we can have a single class to handle dialogs
    This class is intended to load different dialog files and presents a general 
    button method that can be overritten.
    """
    def __init__(self, dialog_name: str, *args, **kwargs):
        super(generalDialog, self).__init__()  # Inherit from QDialog
        self.basedir = os.getcwd()
        if args:
            self.communicate = args[0]
        ui_dir = os.path.join(self.basedir, "ui", dialog_name)
        uic.loadUi(ui_dir, self)
        self.okButton()
        self.show()

    def okButton(self):
        """
        General OK Button box
        Designed to be overwritten where necessary for child classes
        """
        # Definitions here, widgets, variables etc.,
        self.ok_button = self.findChild(QDialogButtonBox, 'okButtonBox')
        self.ok_button.clicked.connect(self.close)

class aboutDialog(generalDialog):
    def __init__(self):
        ui_name = "aboutDialog.ui"
        super(aboutDialog, self).__init__(ui_name)

        self.metadata = self.findChild(
            QLabel, 'appMetaDataLabel')  # Print file location
        self.metadata.setText(self.__get_metadata())

    def __get_metadata(self):
        """
        Get metadata from python module

        Returns:
            metadat_str (str): String metadata for the application
        """
        author = Author(**magic_metadata)
        attrs = vars(author)
        metadata_str = "".join("%s %s\n" % item for item in attrs.items())
        return metadata_str

class noFileDialog(QDialog):
    def __init__(self):
        super(noFileDialog, self).__init__()
        QMessageBox.warning(
            self,
            "Warning!",
            "No file selected! Please select a file first.",
            buttons=QMessageBox.StandardButton.Close,
            defaultButton=QMessageBox.StandardButton.Close,
        )

class helpDialog(generalDialog):
    """
    Open up the README.txt or README.md as a dialog box
    """
    def __init__(self):
        ui_name = "helpDialog.ui"
        super(helpDialog, self).__init__(ui_name) # Inherit from QDialog
        
        self.readme = self.findChild(
            QTextBrowser, 'textBrowser')  # Print file location
        self.readme.setPlainText(self.__get_readme())
    
    def __get_readme(self):
        """
        Get the readme data
        """
        basedir = self.basedir
        readme = os.path.join(basedir, "README.md")
        with open(readme, "r") as text:
            contents = text.read()
        return contents

class errorDialog(QMessageBox):
    """
    Use inbuilt critical message box instead
    """
    def __init__(self, msg:str):
        super(errorDialog, self).__init__()

        QMessageBox.critical(
            self,
            "Error",
            msg,
            buttons=QMessageBox.StandardButton.Close,
            defaultButton=QMessageBox.StandardButton.Close,
        )

class outputFileDialog(generalDialog):
    """
    Error dialog that gets populated based on error that is raised and captured
    """
    def __init__(self, output_file_name:str):
        """
        Start up the error dialog and pass in any catched errors as needed to present

        Args:
            output file name (str): name of output file
        """
        name = "outputFileDialog.ui"
        super(outputFileDialog, self).__init__(name)
        print(output_file_name)
        self.output_file_name = output_file_name
        self.set_output_file_name()
    
    def set_output_file_name(self):
        """_summary_
        """
        output_file_name = self.output_file_name
        self.outputFileOpen = self.findChild(
            QLabel, 'outputFileDialogLabel')  # Print file location
        self.open_file_options(output_file_name)

        self.okButton()
        self.output_file_name = output_file_name
    
    def okButton(self):
        """
        Override the parent class method
        """
        self.buttonBox = self.findChild(QDialogButtonBox, 'buttonBox')
        self.buttonBox.accepted.connect(self.__open_output_file) #(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def open_file_options(self, output_file_name:str):
        """
        Open file options and display output file type

        Args:
            output_file_name (str): name of output file
        """
        self.outputFileOpen.setText(f"This will open the file {output_file_name}. Continue?")
    
    def __open_output_file(self):
        """
        Open output file if button is clicked

        Args:
            output_file_name (str): file name to be opened
        """
        if sys.platform == "win32":
            os.startfile(self.output_file_name) # os.startfile() works for windows, need subprocess for linux
        elif sys.platform == "darwin": # MacOS
            sp.call(["open", self.output_file_name])
            self.accept

class metaDataDialog(generalDialog):
    """
    Print out additional metadata as a separate dialog
    re-use the help dialog ui
    """
    def __init__(self, metadata):
        name = "seeMoreDialog.ui"
        super(metaDataDialog, self).__init__(name) # Inherit from QDialog

        self.readme = self.findChild(
            QTextBrowser, 'textBrowser')  # Print file location
        self.readme.setPlainText(metadata)

class executionFinishedDialog(QDialog):
    """
    Dialog for when execution finished
    
    Re-uses the noFileDialog UI
    """
    def __init__(self, communicate = dialogCommunicate()):
        super(executionFinishedDialog, self).__init__() # Inherit from QDialog
        # load ui for the main window
        basedir = os.getcwd()
        ui_dir = os.path.join(basedir, "ui", "executionFinishedDialog.ui")
        uic.loadUi(ui_dir, self)

        self.di_communicate = communicate
        self.ok_button = self.findChild(QDialogButtonBox, 'okButtonBox')
        self.ok_button.clicked.connect(self.__close)
        self.di_communicate.closure.connect(self.__temp_slot)


        self.noFile = self.findChild(
            QLabel, 'executionFinishedDialogLabel')  # Print file location
        self.noFile.setText("Execution Finished!")
        # Show the app
        self.show()
    
    def __temp_slot(self):
        """
        Temporaray slot to be filled
        """
        print("Temp slot")
    
    def __close(self):
        """
        helper method to close dialog and emit a signal
        """
        self.di_communicate.closure.emit(1)
        #self.closure.emit(1)
        self.close
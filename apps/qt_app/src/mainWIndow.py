import sys
import os
import logging
import subprocess as sp
import json

from PyQt6.QtWidgets import (QFileDialog)
from PyQt6.QtGui import QAction
from PyQt6 import uic
from PyQt6.QtCore import *

from .dialogs import (aboutDialog, noFileDialog, helpDialog, 
                     errorDialog, outputFileDialog, metaDataDialog)
from .signals import PageWindow
from .splashWindow import splashWindow

def except_hook(cls, exception, traceback):
    """
    Capture exceptions and prevent GUI from auto closing upong errors
    Args:
        exception (_type_): _description_
        traceback (_type_): _description_
    """
    sys.__excepthook__(cls, exception, traceback)

basedir = os.getcwd()
logging.basicConfig(filename=os.path.join(basedir,"logs","ERR.log"),
                    format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.ERROR)
logger = logging.getLogger(__name__)

class MainWindow(PageWindow):
    def __init__(self):

        super(MainWindow, self).__init__()  # Inherit from QMainWindow
        # self.init_splash_window()
        self.init_main_window()

    def init_main_window(self):
        """
        Method to initialize the main window
        """
        basedir = os.getcwd() # os.path.dirname(sys.argv[0])
        ui_dir = os.path.join(basedir, 'ui', 'mainWindow.ui')
        
        ui = uic.loadUi(ui_dir, self)
        

        # Menu bar
        self.about = self.findChild(QAction, 'actionAbout')
        self.about.triggered.connect(self.__open_about_window)

        self.help = self.findChild(QAction, 'actionHelp')
        self.help.triggered.connect(self.__open_help_window)

        self.open_file_menubar = self.findChild(QAction, 'actionOpen')
        self.open_file_menubar.triggered.connect(self.__select_file_clicker)

        self.close_file_menubar = self.findChild(QAction, 'actionClose')
        self.close_file_menubar.triggered.connect(self.__close_file_action)

        self.error_menubar = self.findChild(QAction, 'actionOpen_Errors_Folder')
        self.error_menubar.triggered.connect(self.__open_error_folder)

        self.exit_application_menubar = self.findChild(QAction, 'actionExit')
        self.exit_application_menubar.triggered.connect(self.close) # Terminate

        # Have memory of last opened folder here that does not persist across open and closing of application
        self.last_opened_folder = ''

        self.worker = None

        # Show the app
        ui.show()

    def make_handleButton(self, button):
        """
        This button is to allow for switching between pages

        Args:
            button (function): Function that emits a signal to be captured
        """
        def handleButton():
            if button == "speechButton":
                self.goto("speech")
        return handleButton

    def _get_file_formats(self):
        """
        Get the available file formats for checking stuff
        """
        basedir = os.path.dirname(sys.argv[0]) #os.path.dirname(sys.argv[0])
        print(basedir)
        creds_dir = os.path.join(basedir, "config", "formats.json")
        with open(creds_dir, "r") as f:
            config = json.load(f)
            self.audio_formats = config['Audio']
            self.text_formats = config['Text']
            self.csv_formats = config['Excel']

    def __select_file_clicker(self):
        """
        When triggered, allow user to select files of a specific type only
        """
        # Let dialog be called open file, open user to cwd and allow wav files.
        # Upon first execution we will populate the opened folder location as default
        # We then save memory of the chosen location to be repurposed
        filters = "Input files(*.csv)" # add CSV later
        self._get_file_formats() # Set the formats here in memory so we can trace as we need
        if len(self.last_opened_folder) == 0:
            fname = QFileDialog.getOpenFileName(
                self, "Open File", "/", filters
            )
        else:
                fname = QFileDialog.getOpenFileName(
                self, "Open File", self.last_opened_folder, filters
            )
        # Store the filename in memory as string so we can load it for transcription
        self.fname = fname[0]
        self.last_opened_folder = self.fname.rsplit('/',1)[0]

        # Print out the file location on load to the gui
        if len(self.fname) <=0:
            self.__no_file_dialog()
        else:
            self.file_locations.setText(f"{self.fname}")
            self.file_locations_2.setText(f"{self.fname.split('/')[-1]}")
    
    def __format_checker(self):
        """
        check format of input and do something as needed based off it
        """
        # If text based input, process it as we need
        if self.fname.rsplit(".", 1)[1] in self.text_formats:
            self.__read_text()
        elif (self.fname.rsplit(".", 1)[1] in self.audio_formats) and (self.fname.rsplit(".", 1)[1] != "wav"):
            self._reformat()

    def __open_about_window(self):
        """
        Open the About Dialog Window that contains program metadata
        """
        aboutWindow = aboutDialog()
        aboutWindow.exec()

    def __open_help_window(self):
        """
        Open the Help Dialog Window that will display the readme
        """
        helpWindow = helpDialog()
        helpWindow.exec()

    def __close_file_action(self):
        """
        Removes the filename from memory so we cannot transcribe
        """
        if len(self.fname) > 0:
            self.__clear_labels()
            self.fname = ""
            self.file_locations.setText("")
            self.file_locations_2.setText("")
            self.file_metadata.setText("")
    
    def __open_error_folder(self):
        """
        Open folder to errors
        """
        basedir = os.getcwd() 
        error_dir = os.path.join(basedir,"logs")
        if sys.platform == "win32":
            os.startfile(error_dir) # os.startfile() works for windows, need subprocess for linux
        elif sys.platform == "darwin": # MacOS
            sp.call(["open", error_dir])
    
    def __open_output_file_dialog(self):
        """
        Output file dialog box
        """
        output_file_dialog = outputFileDialog(self.output_name)
        output_file_dialog.exec()
    
    def __open_output_folder(self):
        """
        Open output folder based off output filename
        """
        output_folder_path = self.output_name.rsplit('/', 1)[0] # e.g. "folder/path/file.format" -> ["folder/path", "file.format"][0] -> ["folder/path"]

        if sys.platform == "win32":
            os.startfile(output_folder_path) # os.startfile() works for windows, need subprocess for linux
        elif sys.platform == "darwin": # MacOS
            sp.call(["open", output_folder_path])
    
    def __open_metadata_dialog(self):
        """
        Open separate dialog box to show more indepth metadata
        """
        if len(self.fname)>0:
            metaDataWindow = metaDataDialog(self.file_metadata_verbose.extra_info)
            metaDataWindow.exec()
        else:
            self.__no_file_dialog()

    def __no_file_dialog(self):
        """
        No file dialog box
        """
        no_file_dialog = noFileDialog()
    
    def __clear_labels(self):
        """
        Clear all labels if new execution
        """
        self.output_label.setPlainText("")
        self.execution_start.setText("") 
        self.total_execution_time.setText("")
        self.file_output_location.setText("")
    
    
    def __error_dialog(self, msg):
        """
        One liner function that needs to be called where errors will be caught
        Args:
            msg (str): Error message
        """
        # error_dialog = errorDialog(msg)
        # error_dialog.exec()
        errorDialog(msg)

    def __hide_settings(self):
        """
        Method to dynamically hide the navigation bar
        """
        self.sidebar_counter += 1
        if self.sidebar_counter % 2 == 0:
            self.settings_group_box.setHidden(1)
            self.execution_details_group_box.setHidden(1)
            self.file_metadata.setHidden(1)
            self.metadata_label.setHidden(1)
            self.sidebar_divider.move(130, 0)
            self.sidebar_background.setFixedWidth(130)
        else:
            self.settings_group_box.setVisible(1)
            self.execution_details_group_box.setVisible(1)
            self.file_metadata.setVisible(1)
            self.metadata_label.setVisible(1)
            self.sidebar_divider.move(290, 0)
            self.sidebar_background.setFixedWidth(290)

    def hide_startup(self):
        """
        Method to settings on startup
        """
        self.__hide_settings()
        self.output_groupbox.setHidden(1)
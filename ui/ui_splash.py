# Form implementation generated from reading ui file 'c:\Users\k66gu\Documents\auto_ml\ui\splash.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(350, 150)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Background = QtWidgets.QFrame(parent=Form)
        self.Background.setStyleSheet("#Background{\n"
"    background-color: qlineargradient(spread:repeat, x1:1, y1:1, x2:0, y2:0, stop:0 rgba(86, 0, 74, 255), stop:1 rgba(5, 0, 82, 255));\n"
"border: none;\n"
"}")
        self.Background.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.Background.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.Background.setObjectName("Background")
        self.progressBar = QtWidgets.QProgressBar(parent=self.Background)
        self.progressBar.setGeometry(QtCore.QRect(0, 0, 350, 150))
        self.progressBar.setStyleSheet("QProgressBar{\n"
"    text-align:center;\n"
"    color:transparent;\n"
"    background-color: qlineargradient(spread:repeat, x1:1, y1:1, x2:0, y2:0, stop:0 rgba(86, 0, 74, 255), stop:1 rgba(5, 0, 82, 255));\n"
"    border:none;\n"
"}\n"
"QProgressBar:chunk{\n"
"    background-color: qradialgradient(spread:pad, cx:0.6, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:1 rgba(255, 255, 255, 15));\n"
"    border:none;\n"
"}")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label = QtWidgets.QLabel(parent=self.Background)
        self.label.setGeometry(QtCore.QRect(40, 40, 261, 81))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("#label {\n"
"    color: rgb(255, 255, 255);\n"
"}")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.Background)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "MAGIC ML"))

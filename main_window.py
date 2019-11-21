# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 747)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 10, 491, 681))
        self.textBrowser.setObjectName("textBrowser")
        self.cmbox_model_select = QtWidgets.QComboBox(self.centralwidget)
        self.cmbox_model_select.setGeometry(QtCore.QRect(630, 70, 271, 22))
        self.cmbox_model_select.setObjectName("cmbox_model_select")
        self.cmbox_data_dir = QtWidgets.QComboBox(self.centralwidget)
        self.cmbox_data_dir.setGeometry(QtCore.QRect(630, 30, 271, 22))
        self.cmbox_data_dir.setObjectName("cmbox_data_dir")
        self.in_num_classes = QtWidgets.QLineEdit(self.centralwidget)
        self.in_num_classes.setGeometry(QtCore.QRect(630, 100, 113, 22))
        self.in_num_classes.setObjectName("in_num_classes")
        self.in_batch_size = QtWidgets.QLineEdit(self.centralwidget)
        self.in_batch_size.setGeometry(QtCore.QRect(630, 120, 113, 22))
        self.in_batch_size.setObjectName("in_batch_size")
        self.in_epoches = QtWidgets.QLineEdit(self.centralwidget)
        self.in_epoches.setGeometry(QtCore.QRect(630, 150, 113, 22))
        self.in_epoches.setObjectName("in_epoches")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(760, 100, 161, 80))
        self.groupBox.setObjectName("groupBox")
        self.in_rdbtn_Feature = QtWidgets.QRadioButton(self.groupBox)
        self.in_rdbtn_Feature.setGeometry(QtCore.QRect(10, 20, 181, 20))
        self.in_rdbtn_Feature.setChecked(True)
        self.in_rdbtn_Feature.setObjectName("in_rdbtn_Feature")
        self.in_rdbtn_Fine_tune = QtWidgets.QRadioButton(self.groupBox)
        self.in_rdbtn_Fine_tune.setGeometry(QtCore.QRect(10, 40, 95, 20))
        self.in_rdbtn_Fine_tune.setObjectName("in_rdbtn_Fine_tune")
        self.btn_train = QtWidgets.QPushButton(self.centralwidget)
        self.btn_train.setGeometry(QtCore.QRect(550, 280, 93, 28))
        self.btn_train.setObjectName("btn_train")
        self.in_chbox_pretrained = QtWidgets.QCheckBox(self.centralwidget)
        self.in_chbox_pretrained.setGeometry(QtCore.QRect(610, 210, 261, 20))
        self.in_chbox_pretrained.setChecked(True)
        self.in_chbox_pretrained.setObjectName("in_chbox_pretrained")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Mode"))
        self.in_rdbtn_Feature.setText(_translate("MainWindow", "Fearture extraction"))
        self.in_rdbtn_Fine_tune.setText(_translate("MainWindow", "Fine Tuning"))
        self.btn_train.setText(_translate("MainWindow", "Train"))
        self.in_chbox_pretrained.setText(_translate("MainWindow", "Use Pretrained Model on Image Net"))

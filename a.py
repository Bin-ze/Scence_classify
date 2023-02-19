# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'a.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import os
import json
import sys
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif']=['SimHei']
#plt.rcParams['axes.unicode_minus']=False

from model import resnet34,resnet101
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.init_detection()
        self.init_logo()
        self.init_slots()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_video.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.pushButton_video, 0, QtCore.Qt.AlignHCenter)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "开放场景识别系统"))
        self.pushButton_camera.setText(_translate("MainWindow", "打开图片"))
        self.pushButton_video.setText(_translate("MainWindow", "场景检测"))
        self.pushButton.setText(_translate("MainWindow", "退出"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
    def init_slots(self):
        self.pushButton_camera.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.detection)
    def init_logo(self):
        pix=QtGui.QPixmap('a.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)
    def button_image_open(self):
        self.img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.label.setScaledContents(True)
        self.label.setPixmap(QtGui.QPixmap(self.img_name))

    def init_detection(self):
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        self.class_indict = json.load(json_file)
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.model = resnet34(num_classes=51).to(self.device)
        weights_path = "./model_51.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def detection(self):
        img = Image.open(self.img_name)
        plt.axis('off')
        plt.imshow(img)
        img = self.data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        self.model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print_res = "class: {} ".format(self.class_indict[str(predict_cla)])
        plt.title(print_res)
        plt.savefig(self.img_name.split('.')[0]+'_new.jpg')
        self.label.setPixmap(QtGui.QPixmap(self.img_name.split('.')[0]+'_new.jpg'))
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())




import os

import sys
print(os.getcwd())
import pprint
os.system('pyuic5 main_window.ui > main_window.py')
from PyQt5.QtWidgets import    QMainWindow ,QApplication
from main_window import Ui_MainWindow
import utils_2 as util
import pytorch_funcs as TR




class AppWindow(QMainWindow):
    def __init__(self):
        super(AppWindow,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        yaml=os.path.join(os.getcwd(),'config.yml')
        print(yaml)
        self.cfg=util.import_yaml(yaml)

        util.Qlogging(self.ui.textBrowser, 'The config File is loaded \n',"g")
        cfg_str=pprint.pformat(self.cfg)
        # print(cfg_str)
        util.Qlogging(self.ui.textBrowser,cfg_str,'b')
        self.Ui_config_()


    def Ui_config_(self):
        self.ui.btn_train.clicked.connect(self.btn_train)

        self.ui.cmbox_data_dir.addItems(self.cfg['data_dir'])
        self.ui.cmbox_model_select.addItems(self.cfg['model_names'])

        self.ui.in_num_classes.setText(str(self.cfg['num_classes']))
        self.ui.in_num_classes.setEnabled(False)

        self.ui.in_batch_size.setText(str(self.cfg['batch_size']))
        self.ui.in_epoches.setText(str(self.cfg['num_epochs']))



    def update_cfg(self):
        self.new_cfg={}
        self.new_cfg.update({'model_name': self.ui.cmbox_model_select.currentText()})
        self.new_cfg.update({'num_classes': self.ui.in_num_classes.text()})
        self.new_cfg.update({'batch_size': self.ui.in_batch_size.text()})
        self.new_cfg.update({'num_epochs': self.ui.in_epoches.text()})
        self.new_cfg.update({'feature_extract': self.ui.in_rdbtn_Feature.isChecked()})
        self.new_cfg.update({'use_pretrained': self.ui.in_chbox_pretrained.isChecked()})
        print(self.new_cfg)




    def btn_train(self):
        self.update_cfg()
        cfg=self.new_cfg
        model_ft, input_size = TR.initialize_model(cfg['model_name'],
                                                   2,
                                                   cfg['feature_extract'],
                                                   use_pretrained=cfg['use_pretrained'])
        util.Qlogging(self.ui.textBrowser, 'The Model is loaded\n', "r")
        print(model_ft)


app=QApplication(sys.argv)
win=AppWindow()
win.show()
sys.exit(app.exec())
import os

import sys
# print(os.getcwd())
import pprint
os.system('pyuic5 main_window.ui > main_window.py')
from PyQt5.QtWidgets import    QMainWindow ,QApplication
from main_window import Ui_MainWindow
import utils_2 as util
import pytorch_funcs as TR


from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim


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
        self.new_cfg.update({'data_dir': self.ui.cmbox_data_dir.currentText()})
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
                                                   int(cfg['num_classes']),
                                                   cfg['feature_extract'],
                                                   use_pretrained=cfg['use_pretrained'])
        util.Qlogging(self.ui.textBrowser, 'The Model is loaded\n', "r")
        print(model_ft)

        data_transforms = TR.Data_Augmrntation_Normalization(input_size)

        print("Initializing Datasets and Dataloaders...")
        #
        # # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(cfg['data_dir'], x), data_transforms[x]) for x in
                          ['train', 'val']}
        # # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=int(cfg['batch_size']), shuffle=True, num_workers=4) for x
            in ['train', 'val']}
        #
        # # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model_ft = model_ft.to(device)

        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if cfg['feature_extract']:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = TR.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=int(cfg['num_epochs']),
                                     is_inception=(cfg['model_name'] == "inception"))

if __name__ == '__main__':

    app=QApplication(sys.argv)
    win=AppWindow()
    win.show()
    sys.exit(app.exec())
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


#multi Thread programming
from PyQt5.QtCore import  QObject ,pyqtSignal
from PyQt5.QtCore import  QRunnable ,QThreadPool, pyqtSlot
import traceback



class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` data returned from processing, anything
    progress
        `int` indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class AppWindow(QMainWindow):
    def __init__(self):
        super(AppWindow,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        yaml=os.path.join(os.getcwd(),'config.yml')
        print(yaml)
        self.cfg=util.import_yaml(yaml)

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        util.Qlogging(self.ui.textBrowser, 'The config File is loaded \n',"g")
        cfg_str=pprint.pformat(self.cfg)
        # print(cfg_str)
        util.Qlogging(self.ui.textBrowser,cfg_str,'b')
        self.Ui_config_()


    def Ui_config_(self):
        self.ui.btn_train.clicked.connect(self.btn_train)
        self.ui.btn_designNet.clicked.connect(self.btn_design_model)

        self.ui.cmbox_data_dir.addItems(self.cfg['data_dir'])
        self.ui.cmbox_model_select.addItems(self.cfg['model_names'])

        self.ui.in_num_classes.setText(str(self.cfg['num_classes']))
        self.ui.in_num_classes.setEnabled(False)

        self.ui.in_batch_size.setText(str(self.cfg['batch_size']))
        self.ui.in_epoches.setText(str(self.cfg['num_epochs']))

    def btn_design_model(self):
        model_cfg=util.import_yaml('.\models\model_001.yml')
        net=TR.AliNet(model_cfg['model_layers'])
        util.Qlogging(str(net.modules()),'r')

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

        # KK = TR.AliNet(2)
        # model_ft= nn.Sequential(model_ft,nn.Linear(2,200),
        #                         nn.Linear(200,2),KK)





        print(model_ft)


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
        # model_ft, hist = TR.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=int(cfg['num_epochs']),
        #                              is_inception=(cfg['model_name'] == "inception"))
        worker_train= Worker(TR.train_model,model_ft,dataloaders_dict,criterion,optimizer_ft, num_epochs=int(cfg['num_epochs']),is_inception=(cfg['model_name'] == "inception"))
        self.threadpool.start(worker_train)

if __name__ == '__main__':

    app=QApplication(sys.argv)
    win=AppWindow()
    win.show()
    sys.exit(app.exec())
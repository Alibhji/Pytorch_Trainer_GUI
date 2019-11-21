import os

from  PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import  QTableWidgetItem

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import  pickle
import numpy as np



class utils():
    def __init__(self,ui):
        ui.ui.textBrowser.setText("The program is started.")
        self.color()
        self.ui=ui

    def color(self):

        self.black = QColor(0, 0, 0)
        self.red = QColor(255, 0, 0)
        self.green = QColor(0, 100, 0)


    def logging(self,message=" ", type='info'):
        if(type=='info'):
            self.ui.ui.textBrowser.setTextColor(self.green)
            self.ui.ui.textBrowser.append(message)
        if(type=='red'):
            self.ui.ui.textBrowser.setTextColor(self.red)
            self.ui.ui.textBrowser.append(message)
            self.ui.ui.textBrowser.setTextColor(self.black)


    def plotting(self,ui):
        fig = plt.figure(figsize=(9, 15))
        gs = gridspec.GridSpec(nrows=4, ncols=4)

        for i in range(4):
            for j in range(4):
                ax = fig.add_subplot(gs[j, i])
                ax.imshow(list(ui.image_datasets['train'])[i][0].numpy()[1,:,:])
            # ax.set_title(title[i])
        plt.show()

    def check_dir(self,name, root='./' , create_dir=None):
        file = os.path.join(root,name)

        exist= [True,False][os.path.exists(file)]
        if create_dir and exist :
            os.makedirs(file)



    def fill_out_table(self, dict_data):
        # print(dict_data)

        self.ui.ui.tableWidget.setRowCount(len(dict_data))
        for row, row_data in enumerate(sorted(dict_data.items())):
            # for col , data in enumerate(row_data[1].items()):
            #     self.ui.ui.tableWidget.setItem(row,col,QTableWidgetItem(str(data[1])))
            self.ui.ui.tableWidget.setItem(row,0,QTableWidgetItem(str(row_data[1]['name'])))
            if(row_data[1]['trained']):
                self.ui.ui.tableWidget.item(row, 0).setForeground(self.green)
            else:
                self.ui.ui.tableWidget.item(row, 0).setForeground(self.red)

    def fill_out_table_2(self, dict_data):
        lenK = len(self.ui.config['models_outputs'])
        lenO = len(self.ui.config['models_kernels'])
        # print(dict_data)

        self.ui.ui.tableWidget_2.setRowCount   (lenK)
        self.ui.ui.tableWidget_2.setColumnCount(lenO)

        self.ui.ui.tableWidget_2.setHorizontalHeaderLabels(['k='+str(i) for i in self.ui.config['models_kernels']])
        self.ui.ui.tableWidget_2.setVerticalHeaderLabels  ([str(i) for i in self.ui.config['models_outputs']])
        # self.ui.ui.tableWidget_2.setHorizontalHeaderLabels(['1','2','3'])


        for row, row_data in enumerate(sorted(dict_data.items())):
            # for col , data in enumerate(row_data[1].items()):
            #     self.ui.ui.tableWidget.setItem(row,col,QTableWidgetItem(str(data[1])))
            rr = int (row /lenO)
            cc = row % lenO
            self.ui.ui.tableWidget_2.setItem(rr, cc, QTableWidgetItem(str(row_data[1]['code']).split('_')[0]))


            if (row_data[1]['trained']):
                self.ui.ui.tableWidget_2.item(rr, cc).setForeground(self.green)
            else:
                self.ui.ui.tableWidget_2.item(rr, cc).setForeground(self.red)


    def save_object(self, path , object):

        with open(path, 'wb') as uiFile:
            # Step 3
            pickle.dump(object, uiFile)


    def load_object(self, path ):

        with open(path, 'rb') as uiFile:
            # Step 3
            object = pickle.load(uiFile)
            return object

    from mpl_toolkits import  mplot3d
    def plot_3d_(self, path):

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        def load(p):
            with open(p, 'rb') as uiFile:
                # Step 3
                object = pickle.load(uiFile)
                return object

        data=load(path)

        modules = [i for i in list(data.keys()) if i.startswith('Module')]

        k=data['params']['models_kernels']
        ch = data['params']['models_outputs']

        choose = self.ui.ui.in_cmbox_3dPlot.currentText()

        self.ui.ui.in_cmbox_3dPlot.clear()
        for item in ch:
            self.ui.ui.in_cmbox_3dPlot.addItem(str(item))


        print('combo box:',int(choose))



        Z=np.random.rand(len(ch),len(k))
        # Z = np.zeros((len(ch), len(k)))
        Z1, Z2, Z3, Z4 , names =[] ,[], [], [] ,[]
        indx=0
        for i in k:
            # for j in [ch[0]]:
            j = int(choose)
            Module_name = 'Module_{:02d}L_{:02d}ich_{:003d}och_{:02d}k_{:02d}p'.format(3,
                                                                                       3,
                                                                                       j,
                                                                                       i,
                                                                                       int(i / 2))
            l = data[Module_name]['loss']['loss_bce_train']
            samples = [kk[0] for kk in l]
            # epoche = [k[1] for k in l]
            loss1  = [kk[2] for kk in l]
            loss2  = [kk[3] for kk in l]
            loss3  = [kk[4] for kk in l]
            lr     = [kk[5] for kk in l]


            # print('sample', sample)
            # print('loss1', loss1)
            # print('loss2', loss2)
            # print('loss3', loss3)
            # print('lr', loss3)

            Z1.append(loss1)
            Z2.append(loss3)
            Z3.append(loss3)
            Z4.append(lr)
            names.append('{}_{:003d}ch_{:02d}k'.format(indx, j,i))
            indx+=1


        print('shape:',np.array(Z1).shape)


        # X, Y = np.meshgrid(list(range(np.array(Z1).shape[1])), list(range(np.array(Z1).shape[0])))
        X, Y = np.meshgrid(samples, list(range(np.array(Z1).shape[0])))

        print('samples:', X,Y)
        ind_x=list(range(np.array(Z1).shape[1]))


        Z = np.array(Z1)[Y, ind_x]

        ax = plt.axes(projection='3d')

        ax.view_init(60, 45)
        plt.title('loss changing b', fontdict=font)
        # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
        plt.xlabel('Module_Name', fontdict=font)
        # plt.yticks(list(range(np.array(Z1).shape[1])), str(samples).replace('[','' ).replace(']','').split(','))
        plt.xticks(list(range(np.array(Z1).shape[0])), str(names).replace('[','' ).replace(']','').split(','))
        print(samples)
        plt.ylabel('Sample numbers', fontdict=font)
        # ax.scatter3D(Y, X, Z, cmap='Greens')
        plt.style.use('classic')
        # ax.plot_wireframe(Y, X, Z, color='black')
        # ax.contour3D(Y, X, Z, 50, cmap='Blues')
        ax.plot_surface(Y, X, Z, rstride=1, cstride=1000, color='w', shade=False, lw=.5)

        plt.show()



if __name__ == '__main__':

    plot_3d_('C:\\Users\\alibh\\Desktop\\My_Qt\\Paper_2_PyQT_Pytorch\\designed_modules\\All_Results.losses')

import joypy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# from bokeh.plotting import figure,show
# import chart_studio.plotly as py
# import cufflinks as cf

# import altair as alt



from matplotlib import cm

def plot1(loss_data):
    # print(str(loss_data))


    df_train=pd.DataFrame(loss_data['loss_bce_train'])

    fig, (ax1, ax2) = plt.subplots(2,1)

    df_train.columns = ['Sample', 'epoch', 'bce', 'loss_defined', 'total', 'lr']
    # df_train[["bce", 'loss_defined', 'total']]

    # fig, axes= joypy.joyplot(df_train ,column=["bce", 'loss_defined', 'total'])

    # # ax.imshow(df_train[["bce", 'loss_defined', 'total']].plot.kde())
    # df_train[["bce", 'loss_defined', 'total']].plot.kde(ax=ax1)
    df_train[["bce", 'loss_defined', 'total']].plot.kde(ax=ax2)
    df_train.plot(x='Sample',y=['bce', 'loss_defined', 'total'],ax=ax1)
    # fig.savefig('./test.png')



    plt.show()

    # p = figure(width=500,height=500)
    # p.line(x=df_train['Sample'] ,y=df_train['bce'])
    # show(p)



    
    # grid_size = (3, 2)
    # hosts_to_fmt = []
    # fig = plt.figure(figsize=(15, 12))

    
    # Place A Title On The Figure

    #     # fig.show()
    # joypy.joyplot(df_train)
    # print(df_train)

    # alt.Chart(df_train).mark_point()


# plot1()

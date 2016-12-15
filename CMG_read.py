# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:15:01 2016

@author: eric
"""
import numpy as np
from pylab import *
pgf_with_custom_preamble = {
    "font.family": "serif",   # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
}

matplotlib.rcParams.update(pgf_with_custom_preamble)
import matplotlib.pyplot as plt
class CMG_read():
    def __init__(self,CMG_file_path):
        data=np.loadtxt(CMG_file_path,skiprows=4)
        self.x=data[:,0]-400; self.y=(data[:,1]+400); self.pressure=(data[:,2]/1000)
        self.num_ele=int(np.sqrt(len(self.x)))
    def p_function(self,x_vec,y_vec):
        x_index=np.round((x_vec+400)/800*(self.num_ele-1)).astype(int)
        y_index=(self.num_ele-1)-np.round((y_vec+400)/800*(self.num_ele-1)).astype(int)
#        print(x_index,y_index)
        press=self.pressure.reshape(int(self.num_ele),int(self.num_ele))
        return press[y_index,x_index]
        
    def plotter(self):
        X,Y=np.meshgrid(np.linspace(-400,400,200),np.linspace(-400,400,200))
        Z=self.p_function(X,Y)
        ticks=np.linspace(min(self.pressure),max(self.pressure),5)

        fig,ax=plt.subplots()

        cs=ax.contourf(X,Y,Z,cmap="coolwarm",levels=np.linspace(min(self.pressure),max(self.pressure),20)) 
        cbar=fig.colorbar(cs,ax=ax,ticks=ticks)
#        plt.gca().invert_xaxis()
#        plt.gca().invert_yaxis()

        plt.xticks(np.linspace(-400,400,5))
        plt.yticks(np.linspace(-400,400,5))
#        ax.add_collection(p)
        ax.set_aspect('equal') 
        ax.set_xlim([-400, 400])
        ax.set_ylim([-400, 400])
        props = dict(boxstyle='square', facecolor='white', alpha=1.)
        plt.gcf().subplots_adjust(bottom=0.15)   
        plt.text(0.5, 0.9,'Degrees of freedom= %s' %int(self.num_ele**2), ha='center', va='center', transform=ax.transAxes,bbox=props,fontsize=16)
        plt.xlabel('x [m]',fontsize=18)
        plt.ylabel('y [m]',fontsize=18)
        plt.tick_params(labelsize=18)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Pressure [MPa]', labelpad=10, size=18)
        
        return plt
        
#CMG_file_path_1='CMG_scurve/linesource_waterflood_20 Pressure Time 1905-01-01.txt'
#CMG_1=CMG_read(CMG_file_path_1)
#plot1=CMG_1.plotter()
#plt.show()
#
#CMG_file_path_2='CMG_scurve/linesource_waterflood_50 Pressure Time 1905-01-01.txt'
#CMG_2=CMG_read(CMG_file_path_2)
#plot2=CMG_2.plotter()
#plt.show()
#
#CMG_file_path_3='CMG_scurve/linesource_waterflood_100 Pressure Time 1905-01-01.txt'
#CMG_3=CMG_read(CMG_file_path_3)
#plot3=CMG_3.plotter()
#plt.show()
#
#CMG_file_path_4='CMG_scurve/linesource_waterflood_200 Pressure Time 1905-01-01.txt'
#CMG_4=CMG_read(CMG_file_path_4)
#plot4=CMG_4.plotter()
#plt.show()
#
#CMG_file_path_5='CMG_scurve/linesource_waterflood_400 Pressure Time 1905-01-01.txt'
#CMG_5=CMG_read(CMG_file_path_5)
#plot5=CMG_5.plotter()
#plt.show()


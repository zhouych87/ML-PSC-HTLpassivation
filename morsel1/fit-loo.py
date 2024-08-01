# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:44:19 2023

@author: DELL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import shap
from multiprocessing.pool import Pool
from functools import partial
import time

from sklearn.feature_selection import SelectPercentile,SelectFromModel 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,KFold,cross_val_score,cross_val_predict,LeaveOneOut
from sklearn.preprocessing import StandardScaler,RobustScaler,MaxAbsScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,make_scorer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge,Lasso,SGDRegressor,BayesianRidge,ARDRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.pipeline import Pipeline
#from xgboost import XGBRegressor
#import lightgbm as lgb
    
def plot1(y_test,y_test_pred,y_pad,r2,r2_pad,tabl9):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '10';fs=12
    plt.figure(figsize=(5,4))
    ax = plt.gca() #s=s控制点的大小,alpha=a控制透明度,edgecolor控制描边
    plt.grid(alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))     
    ax.yaxis.set_major_locator(MultipleLocator(1)) 
    x = np.linspace(0,260,260);y=x
    plt.plot(x,y, 'black')
    plt.scatter(y_test, y_pad,c="royalblue", s=30, marker="+", alpha=0.7,label="E-C: R$^{2}$ = %.2f" % r2_pad)  #cornflowerblue
    plt.scatter(y_test, y_test_pred,c="crimson", s=25, marker="o", alpha=0.7,label= "morSel-1: R$^{2}$ = %.2f" % r2)
    plt.xlim([16, 25]); plt.ylim([16, 25])
    #ax.set_aspect(0.8)
    plt.xlabel("Experimental PCE",fontsize=fs); plt.ylabel("Predicted PCE",fontsize=fs)
    #plt.title("Eads prediction") #change
    plt.legend(fontsize=fs)
    plt.savefig(tabl9+'-loo.png',dpi=300,bbox_inches='tight') #change
    return

if __name__=="__main__":
    st=time.time()
    tablp=pd.read_csv("SDD_M.CSV", header= 0, sep=',')
    tablpd=pd.read_csv("sdd_slcdec.CSV", header= 0, sep=',')
    reg=ARDRegression()
    reg_pad=SVR(C=18, gamma=0.21, epsilon=0.2)
    loo = LeaveOneOut()
    pipe = Pipeline([("scaler", MinMaxScaler()), ("ard", reg)])
    pipe_pad = Pipeline([("scaler", MinMaxScaler()), ("ard", reg_pad)])
    colseq=16
    CPCE=tablp.iloc[:,1]
    PVK=tablp.iloc[:,1:5]
    y_all = tablp.iloc[:,5]
    pad=tablpd.iloc[:,1:12]
    for tabl9 in ['sorted_descriptors_original1.csv']: 
        tablm = pd.read_csv(tabl9, header= 0, sep=',')
        Mor=tablm.iloc[:,:]
        x_all=pd.concat([PVK,Mor],axis=1)    #choose
        x_pad=pd.concat([PVK,pad],axis=1)  
        y_pred = pd.DataFrame(cross_val_predict(pipe, x_all, y_all, cv=loo),index=y_all.index)
        y_pad = pd.DataFrame(cross_val_predict(pipe_pad, x_pad, y_all, cv=loo),index=y_all.index)
        r2 = r2_score(y_all, y_pred)
        r2_pad=r2_score(y_all, y_pad)
        mae=mean_absolute_error(y_all, y_pred)
        mse=mean_squared_error(y_all, y_pred)
        rmse=np.sqrt(mse)
        yy=pd.concat([y_all,y_pred],axis=1)
        pea=yy.corr()
        r=pea.iloc[0,1]
        print("{},{} Average R2: {:.4f} RMSE: {:.4f} r: {:.4f}".format(tabl9,colseq,r2,rmse,r))
        plot1(y_all,y_pred,y_pad,r2,r2_pad,tabl9)
        #ymae=np.absolute(np.array(y_pred).flatten() - np.array(y_all).flatten())
    mse_pad=mean_squared_error(y_all, y_pad)
    rmse_pad=np.sqrt(mse_pad)
    yy=pd.concat([y_all,y_pad],axis=1)
    pea=yy.corr()
    r_pad=pea.iloc[0,1]
    print("pad, Average R2: {:.4f} RMSE: {:.4f} r: {:.4f}".format(r2_pad,rmse_pad,r_pad))
    et=time.time()   
    print("Done in {:.4f} second".format(et-st) )




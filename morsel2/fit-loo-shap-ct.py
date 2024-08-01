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
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,KFold,cross_val_score,LeaveOneOut
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


def fit2(x_all,y_all,i):
    x_train=x_all.drop(i);y_train=y_all.drop(i)
    x_test=x_all.iloc[i].to_frame().T;y_test=y_all.iloc[i]
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    model=reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    explainer = shap.LinearExplainer(model,x_train)
    shap_values = explainer.shap_values(x_test)
    return x_test,y_test_pred, shap_values
    

if __name__=="__main__":
    st=time.time()

    std = MinMaxScaler();reg=ARDRegression()
    tablp=pd.read_csv("SDD_M.CSV", header= 0, sep=',')
    reg=ARDRegression()
    loo = LeaveOneOut()
    pipe = Pipeline([("scaler", MinMaxScaler()), ("ard", reg)]) 
    tabl = pd.read_csv('select_concat-Ayyub-Magic.csv', header= 0, sep=',')
    PVK=tablp.iloc[:,1:5]
    y_all = tablp.iloc[:,5]
    Mor=tabl.iloc[:,:16]
    x_all=pd.concat([PVK,Mor],axis=1)    #choose in three
    lx=[];lyp=[]
    for i in range(len(tablp.index)):
        x_test,y_test_pred, shap_values=fit2(x_all,y_all,i)
        lx.append(x_test)
        lyp.append(y_test_pred)
        if i==0: 
            shap_values_list=np.array(shap_values)
        else:
            shap_values_list=np.vstack((shap_values_list,shap_values))
    rmse=np.sqrt(mean_squared_error(list(y_all), lyp))
    print("Average RMSE: {:.3f}".format(rmse))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure()
    shap.summary_plot(shap_values_list,x_all,show=False) #,max_display=10
    color='cool'
    my_cmap = plt.get_cmap(color)
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(my_cmap)
    ax1 = plt.gcf().get_axes()[0]  # 获取colorbar轴
    ax1.tick_params(labelsize=18)  # 调整colorbar刻度标签的字体大小 16
    ax1.set_xlabel('SHAP value', fontsize=18)  #18
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    cbar = plt.gcf().get_axes()[1]
    cbar.tick_params(labelsize=18) #16
    cbar.set_ylabel('Feature value', fontsize=18)  # 调整colorbar标签的字体大小，同时设置标签文本 16
    plt.tight_layout()
    plt.savefig("Sel-shap-all.png",dpi=300)

    et=time.time()   
    print("Done in {:.4f} second".format(et-st) )




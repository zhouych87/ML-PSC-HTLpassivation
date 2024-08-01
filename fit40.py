# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:44:19 2023

@author: DELL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

if __name__=="__main__":
    st=time.time()
    
    loo = LeaveOneOut()
    reg=ARDRegression()
    pipe = Pipeline([("scaler", MinMaxScaler()), ("ard", reg)])
    tablp=pd.read_csv("SDD_M.CSV", header= 0, sep=',')   
    PVK=tablp.iloc[:,1:5]
    y_all = tablp.iloc[:,5]
    colseq=41
    for tabl9 in ['MorUlist','MorElist','MorIPlist']:   
        tabl = pd.read_csv(tabl9+".CSV", header= 0, sep=',')  
        Mor=tabl.iloc[:,1:colseq]
        x_all=pd.concat([PVK,Mor],axis=1)    #choose in three
        y_pred = pd.DataFrame(cross_val_predict(pipe, x_all, y_all, cv=loo),index=y_all.index)
        r2 = r2_score(y_all, y_pred)
        mse=mean_squared_error(y_all, y_pred); rmse=np.sqrt(mse)
        yy=pd.concat([y_all,y_pred],axis=1); pea=yy.corr(); r=pea.iloc[0,1]
        print("{} Average R2: {:.3f} RMSE: {:.3f} r: {:.3f}".format(tabl9,r2,rmse,r))

    et=time.time()   
    print("Done in {:.4f} second".format(et-st) )




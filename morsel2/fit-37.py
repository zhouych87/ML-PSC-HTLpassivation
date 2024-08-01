# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:44:19 2023

@author: DELL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#import shap
from multiprocessing.pool import Pool
from functools import partial
import time

from sklearn.feature_selection import SelectPercentile,SelectFromModel 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler,RobustScaler,MaxAbsScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,make_scorer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge,Lasso,SGDRegressor,BayesianRidge,ARDRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
#from xgboost import XGBRegressor
#import lightgbm as lgb

def fit1(x_all,y_all,rdn):
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, train_size=0.7,random_state=rdn)
    std = MinMaxScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)    
    reg=ARDRegression() ####
    model=reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    r2_test = r2_score(y_test, y_test_pred)
    y_train_pred = reg.predict(x_train)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test=mean_squared_error(y_test, y_test_pred)
    rmse_test=np.sqrt(mse_test)
    y_test_p=pd.DataFrame(y_test_pred)
    y_test_p.index = y_test.index
    yy=pd.concat([y_test,y_test_p],axis=1)
    pea=yy.corr()
    r_test=pea.iloc[0,1]
    #print("Test R2: {:0.3f} r: {:0.3f} RMSE: {:0.3f}".format(r2_test,r_test,rmse_test))
    y_train_pred = reg.predict(x_train)   
    return r_test, r2_test,r2_train, rmse_test, y_test, y_test_pred, y_train, y_train_pred


def plot1(y_train,y_train_pred,y_test,y_test_pred,lr2,lr2tn,filename):
    plt.rcParams['font.size'] = '10';fs=12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(5,4))
    ax = plt.gca() #s=s控制点的大小,alpha=a控制透明度,edgecolor控制描边
    plt.grid(alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))     
    ax.yaxis.set_major_locator(MultipleLocator(1))
    x = np.linspace(0,260,260);y=x
    plt.plot(x,y, 'black')
    plt.scatter(y_train, y_train_pred,c="gold", edgecolor=None,s=50, marker="o", alpha=0.5,linewidths=0,label="Training set: R$^{2}$ = %.2f" % np.mean(lr2tn))
    plt.scatter(y_test, y_test_pred,c="darkviolet", s=20, marker="+", alpha=0.7,linewidths=0.5, label="Test set: R$^{2}$ = %.2f" % np.mean(lr2))
    #"lightsteelblue","crimson";"gold","darkviolet"
    plt.xlim([16, 25]); plt.ylim([16, 25])
    #ax.set_aspect(0.8)
    plt.xlabel("Experimental PCE",fontsize=fs); plt.ylabel("Predicted PCE",fontsize=fs)
    #plt.title("Eads prediction") #change
    plt.legend(fontsize=fs)
    plt.savefig(filename,dpi=300,bbox_inches='tight') #change
    return

if __name__=="__main__":
    st=time.time()
    tablp=pd.read_csv("SDD_M.CSV", header= 0, sep=',')
    tablpd=pd.read_csv("sdd_slcdec.CSV", header= 0, sep=',')
    CPCE=tablp.iloc[:,1]
    PVK=tablp.iloc[:,1:5]
    pad=tablpd.iloc[:,1:12]
    y_all = tablp.iloc[:,5]
    x_pad=pd.concat([PVK,pad],axis=1)
    lr2tn=[];lr2=[];lrmse=[];lr=[];shap_values_list=[];ly_test=[];ly_train=[];ly_test_pred=[];ly_train_pred=[]
    for i in range(100):
        r_test,r2_test,r2_train,rmse_test, y_test, y_test_pred, y_train, y_train_pred=fit1(x_pad,y_all,i)
        lr.append(r_test); lr2.append(r2_test); lrmse.append(rmse_test)
        lr2tn.append(r2_train)
        ly_test.append(y_test); ly_train.append(y_train)
        ly_test_pred.append(y_test_pred); ly_train_pred.append(y_train_pred)
    yallts= np.array(ly_test).flatten(); yalltn= np.array(ly_train).flatten()
    yallts_p= np.array(ly_test_pred).flatten(); yalltn_p= np.array(ly_train_pred).flatten()  
    plot1(yalltn,yalltn_p,yallts,yallts_p,lr2,lr2tn,'pad-37.png')
    print("pad Average R2: {:.4f} RMSE: {:.4f} r: {:.4f}".format(np.mean(lr2),np.mean(lrmse),np.mean(lr)))
    for tabl9 in ['select_concat.csv']: 
        tablm = pd.read_csv(tabl9, header= 0, sep=',')
        Mor=tablm.iloc[:,:16]
        x_all=pd.concat([PVK,Mor],axis=1)    #choose
        lr2tn=[];lr2=[];lrmse=[];lr=[];shap_values_list=[];ly_test=[];ly_train=[];ly_test_pred=[];ly_train_pred=[]
        for i in range(100):
            r_test,r2_test,r2_train,rmse_test, y_test, y_test_pred, y_train, y_train_pred=fit1(x_all,y_all,i)
            lr.append(r_test); lr2.append(r2_test); lrmse.append(rmse_test)
            lr2tn.append(r2_train)
            #shap_values_list.append(shap_values)
            ly_test.append(y_test); ly_train.append(y_train)
            ly_test_pred.append(y_test_pred); ly_train_pred.append(y_train_pred)
        yallts= np.array(ly_test).flatten(); yalltn= np.array(ly_train).flatten()
        yallts_p= np.array(ly_test_pred).flatten(); yalltn_p= np.array(ly_train_pred).flatten()  
        plot1(yalltn,yalltn_p,yallts,yallts_p,lr2,lr2tn,tabl9+'-37.png')
        print("{} Average R2: {:.4f} RMSE: {:.4f} r: {:.4f}".format(tabl9,np.mean(lr2),np.mean(lrmse),np.mean(lr)))
    et=time.time()   
    print("Done in {:.4f} second".format(et-st) )




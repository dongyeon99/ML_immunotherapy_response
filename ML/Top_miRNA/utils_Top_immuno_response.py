
#### package load ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn import ensemble

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score


#### Data processing ####

#### Train and Test Data Split (Train dataset) ####

def prepare_tr_te(data_folder):
    
    # data laod
    data = pd.read_csv(os.path.join(data_folder,"data.csv"))
    
    train, test = train_test_split(data, train_size=0.8, stratify = data['CTL.flag'], random_state=42)
    
    ### Standard Scaling ###
    x_train = train.iloc[:, 2:len(train.columns)-5]
    x_test = test.iloc[:, 2:len(test.columns)-5]
    
    scaler = StandardScaler()
    x_list = list(x_train.columns)

    x_train_sc = x_train.copy()
    x_train_sc[x_list] = scaler.fit_transform(x_train[x_list])

    x_test_sc = x_test.copy()
    x_test_sc[x_list] = scaler.transform(x_test[x_list])

    print( "Top 1 ( mean(SHAP value) > 0.01 )")
    print("CTL model: Top 3")
    print("Dysfunction model: Top 5")
    print("Exclusion model: Top 12")
    print(" ")
    
    print( "Top 2 ( mean(SHAP value) > 0.02 )")
    print("CTL model: Top 2")
    print("Dysfunction model: Top 4")
    print("Exclusion model: Top 5")
    print(" ")
    
    ### Train dataset ###
    top1_x_train_ctl = x_train_sc[['hsa.mir.155','hsa.mir.150','hsa.mir.4772']]

    top1_x_train_tide = x_train_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183','hsa.mir.155','hsa.mir.345']]

    top1_x_train_dys = x_train_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a','hsa.mir.210']]

    top1_x_train_exc = x_train_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150','hsa.mir.192',
                                   'hsa.mir.141','hsa.mir.155','hsa.mir.200c','hsa.mir.493','hsa.mir.708','hsa.mir.142']]

    top2_x_train_ctl = x_train_sc[['hsa.mir.150','hsa.mir.155']]

    top2_x_train_tide = x_train_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183']]

    top2_x_train_dys = x_train_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a']]

    top2_x_train_exc = x_train_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150']]

    y_train_ctl = train[['CTL.flag']]

    y_train_dys = train[['Dysfunction']]

    y_train_exc = train[['Exclusion']]

    y_train_tide = train[['TIDE']]

    ### Test dataset ###
    top1_x_test_ctl = x_test_sc[['hsa.mir.155','hsa.mir.150','hsa.mir.4772']]

    top1_x_test_tide = x_test_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183','hsa.mir.155','hsa.mir.345']]

    top1_x_test_dys = x_test_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a','hsa.mir.210']]

    top1_x_test_exc = x_test_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150','hsa.mir.192',
                                 'hsa.mir.141','hsa.mir.155','hsa.mir.200c','hsa.mir.493','hsa.mir.708','hsa.mir.142']]

    top2_x_test_ctl = x_test_sc[['hsa.mir.155','hsa.mir.4772']]

    top2_x_test_tide = x_test_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183']]

    top2_x_test_dys = x_test_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a']]

    top2_x_test_exc = x_test_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150']]
    
    y_test_ctl = test[['CTL.flag']]

    y_test_dys = test[['Dysfunction']]

    y_test_exc = test[['Exclusion']]

    y_test_tide = test[['TIDE']]

    return top1_x_train_ctl, top1_x_train_dys, top1_x_train_exc, top1_x_train_tide, top2_x_train_ctl, top2_x_train_dys, top2_x_train_exc, top2_x_train_tide, y_train_ctl, y_train_dys, y_train_exc, y_train_tide, top1_x_test_ctl, top1_x_test_dys, top1_x_test_exc, top1_x_test_tide, top2_x_test_ctl, top2_x_test_dys, top2_x_test_exc, top2_x_test_tide, y_test_ctl, y_test_dys, y_test_exc, y_test_tide



#### Train and Test Data Split (Validation dataset) ####

def prepare_val(data_folder):
    # data laod
    val_data = pd.read_csv(os.path.join(data_folder,"validation_data.csv"))
    ### Standard Scaling ###

    scaler_val = StandardScaler()

    x_val = val_data.iloc[:, 2:len(val_data.columns)-5]

    x_val_list = list(x_val.columns)

    x_val_sc = x_val.copy()
    x_val_sc[x_val_list] = scaler_val.fit_transform(x_val[x_val_list])
    
    ### Top miRNA select ###

    # Top 1 ( mean(SHAP value) > 0.01 )

    top1_x_val_ctl = x_val_sc[['hsa.mir.155','hsa.mir.150','hsa.mir.4772']]

    top1_x_val_tide = x_val_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183','hsa.mir.155','hsa.mir.345']]

    top1_x_val_dys = x_val_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a','hsa.mir.210']]

    top1_x_val_exc = x_val_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150','hsa.mir.192',
                                   'hsa.mir.141','hsa.mir.155','hsa.mir.200c','hsa.mir.493','hsa.mir.708','hsa.mir.142']]


    # Top 2 ( mean(SHAP value) > 0.02 )

    top2_x_val_ctl = x_val_sc[['hsa.mir.150','hsa.mir.155']]

    top2_x_val_tide = x_val_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183']]

    top2_x_val_dys = x_val_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a']]

    top2_x_val_exc = x_val_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150']]


    # Target values
    y_val_ctl = val_data[['CTL.flag']]

    y_val_tide = val_data[['TIDE']]

    y_val_dys = val_data[['Dysfunction']]

    y_val_exc = val_data[['Exclusion']]
    
    return top1_x_val_ctl, top1_x_val_dys, top1_x_val_exc, top1_x_val_tide, top2_x_val_ctl, top2_x_val_dys, top2_x_val_exc, top2_x_val_tide, y_val_ctl, y_val_dys, y_val_exc, y_val_tide 



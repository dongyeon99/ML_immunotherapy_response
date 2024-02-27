
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
    data = pd.read_csv(os.path.join(data_folder,"PCAWG_TIDE_miR.csv"))
    
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
    top1_x_train_ctl = x_train_sc[['hsa-miR-155-3p','hsa-miR-155-5p','hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-4772-3p','hsa-miR-4772-5p']]

    top1_x_train_tide = x_train_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-10a-3p','hsa-miR-10a-5p',
                                    'hsa-miR-183-3p','hsa-miR-183-5p','hsa-miR-155-3p','hsa-miR-155-5p','hsa-miR-345-3p','hsa-miR-345-5p']]

    top1_x_train_dys = x_train_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-183-3p','hsa-miR-183-5p','hsa-miR-150-3p','hsa-miR-150-5p',
                                    'hsa-miR-151a-3p','hsa-miR-151a-5p','hsa-miR-210-3p','hsa-miR-210-5p']]

    top1_x_train_exc = x_train_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-194-3p','hsa-miR-194-5p','hsa-miR-10a-3p','hsa-miR-10a-5p',
                                    'hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-192-3p','hsa-miR-192-5p','hsa-miR-141-3p','hsa-miR-141-5p',
                                    'hsa-miR-155-3p','hsa-miR-155-5p','hsa-miR-200c-3p','hsa-miR-200c-5p','hsa-miR-493-3p','hsa-miR-493-5p',
                                    'hsa-miR-708-3p','hsa-miR-708-5p','hsa-miR-142-3p','hsa-miR-142-3p']]


    top2_x_train_ctl = x_train_sc[['hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-155-3p','hsa-miR-155-5p']]

    top2_x_train_tide = x_train_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-150-3p','hsa-miR-150-5p',
                                    'hsa-miR-10a-3p','hsa-miR-10a-5p','hsa-miR-183-3p','hsa-miR-183-5p']]

    top2_x_train_dys = x_train_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-183-3p','hsa-miR-183-5p',
                                    'hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-151a-3p','hsa-miR-151a-5p']]

    top2_x_train_exc = x_train_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-194-3p','hsa-miR-194-5p',
                                    'hsa-miR-10a-3p','hsa-miR-10a-5p','hsa-miR-150-3p','hsa-miR-150-5p']]


    y_train_ctl = train[['CTL.flag']]

    y_train_dys = train[['Dysfunction']]

    y_train_exc = train[['Exclusion']]

    y_train_tide = train[['TIDE']]

    ### Test dataset ###
    top1_x_test_ctl = x_test_sc[['hsa-miR-155-3p','hsa-miR-155-5p','hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-4772-3p','hsa-miR-4772-5p']]

    top1_x_test_tide = x_test_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-10a-3p','hsa-miR-10a-5p',
                                    'hsa-miR-183-3p','hsa-miR-183-5p','hsa-miR-155-3p','hsa-miR-155-5p','hsa-miR-345-3p','hsa-miR-345-5p']]

    top1_x_test_dys = x_test_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-183-3p','hsa-miR-183-5p','hsa-miR-150-3p','hsa-miR-150-5p',
                                    'hsa-miR-151a-3p','hsa-miR-151a-5p','hsa-miR-210-3p','hsa-miR-210-5p']]

    top1_x_test_exc = x_test_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-194-3p','hsa-miR-194-5p','hsa-miR-10a-3p','hsa-miR-10a-5p',
                                    'hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-192-3p','hsa-miR-192-5p','hsa-miR-141-3p','hsa-miR-141-5p',
                                    'hsa-miR-155-3p','hsa-miR-155-5p','hsa-miR-200c-3p','hsa-miR-200c-5p','hsa-miR-493-3p','hsa-miR-493-5p',
                                    'hsa-miR-708-3p','hsa-miR-708-5p','hsa-miR-142-3p','hsa-miR-142-3p']]


    top2_x_test_ctl = x_test_sc[['hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-155-3p','hsa-miR-155-5p']]

    top2_x_test_tide = x_test_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-150-3p','hsa-miR-150-5p',
                                    'hsa-miR-10a-3p','hsa-miR-10a-5p','hsa-miR-183-3p','hsa-miR-183-5p']]

    top2_x_test_dys = x_test_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-183-3p','hsa-miR-183-5p',
                                    'hsa-miR-150-3p','hsa-miR-150-5p','hsa-miR-151a-3p','hsa-miR-151a-5p']]

    top2_x_test_exc = x_test_sc[['hsa-miR-10b-3p','hsa-miR-10b-5p','hsa-miR-194-3p','hsa-miR-194-5p',
                                    'hsa-miR-10a-3p','hsa-miR-10a-5p','hsa-miR-150-3p','hsa-miR-150-5p']]
    
    y_test_ctl = test[['CTL.flag']]

    y_test_dys = test[['Dysfunction']]

    y_test_exc = test[['Exclusion']]

    y_test_tide = test[['TIDE']]

    return top1_x_train_ctl, top1_x_train_dys, top1_x_train_exc, top1_x_train_tide, top2_x_train_ctl, top2_x_train_dys, top2_x_train_exc, top2_x_train_tide, y_train_ctl, y_train_dys, y_train_exc, y_train_tide, top1_x_test_ctl, top1_x_test_dys, top1_x_test_exc, top1_x_test_tide, top2_x_test_ctl, top2_x_test_dys, top2_x_test_exc, top2_x_test_tide, y_test_ctl, y_test_dys, y_test_exc, y_test_tide



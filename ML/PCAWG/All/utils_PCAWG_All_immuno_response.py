
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

from auto_shap.auto_shap import produce_shap_values_and_summary_plots



#### Data processing ####

#### Train and Test Data Split (Train dataset) ####

def prepare_tr_te(data_folder):
    
    # data laod
    data = pd.read_csv(os.path.join(data_folder,"PCAWG_TIDE_miR.csv"))
    
    train, test = train_test_split(data, train_size=0.8, stratify = data['CTL.flag'], random_state=42)
    

    ### Train dataset ###
    x_train = train.iloc[:, 2:len(train.columns)-5]

    y_train_ctl = train[['CTL.flag']]

    y_train_dys = train[['Dysfunction']]

    y_train_exc = train[['Exclusion']]

    y_train_tide = train[['TIDE']]

    ### Test dataset ###
    x_test = test.iloc[:, 2:len(test.columns)-5]

    y_test_ctl = test[['CTL.flag']]

    y_test_dys = test[['Dysfunction']]

    y_test_exc = test[['Exclusion']]

    y_test_tide = test[['TIDE']]

    return x_train, y_train_ctl, y_train_dys, y_train_exc, y_train_tide, x_test, y_test_ctl, y_test_dys, y_test_exc, y_test_tide



#### SHAP (feature importance) ####

def calculate_shap(model, x_train, model_save_folder, model_list):
    # make folder
    os.mkdir(os.path.join(model_save_folder,"PCAWG_{}_SHAP".format(model_list)))
    # calculate SHAP
    produce_shap_values_and_summary_plots(model=model, x_df=x_train, save_path=os.path.join(model_save_folder,"PCAWG_{}_SHAP".format(model_list)))



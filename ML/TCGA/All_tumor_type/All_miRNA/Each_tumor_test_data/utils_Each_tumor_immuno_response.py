
#### Package load ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score


#### Data processing ####

#### Train and Test Data Split ####

def prepare_tr_te_each(data, data_folder):
    
    tumor_type = pd.read_csv(os.path.join(data_folder,"TIDE_Score_{}_miR_rm_dup.csv".format(data)))
    
    train, test = train_test_split(tumor_type, train_size=0.8, stratify = tumor_type['CTL.flag'], random_state=42)

    ### Train dataset ###
    x_train = train.iloc[:, 1:len(train.columns)-5]

    y_train_ctl = train[['CTL.flag']]

    y_train_dys = train[['Dysfunction']]

    y_train_exc = train[['Exclusion']]

    y_train_tide = train[['TIDE']]

    ### Test dataset ###
    x_test = test.iloc[:, 1:len(test.columns)-5]

    y_test_ctl = test[['CTL.flag']]

    y_test_dys = test[['Dysfunction']]

    y_test_exc = test[['Exclusion']]

    y_test_tide = test[['TIDE']]

    return x_train, y_train_ctl, y_train_dys, y_train_exc, y_train_tide, x_test, y_test_ctl, y_test_dys, y_test_exc, y_test_tide





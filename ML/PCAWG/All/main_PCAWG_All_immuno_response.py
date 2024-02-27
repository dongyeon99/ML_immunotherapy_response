
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

from utils_PCAWG_All_immuno_response import *
from models_PCAWG_All_immuno_response import *


#### Data prepare ####

# Path
data_folder = "/ML_immunotherapy_response/Data/Result/PCAWG/"
model_save_folder = "/ML_immunotherapy_response/ML/PCAWG/All/models/"

# Data load
All = prepare_tr_te(data_folder)


tr_te = ['x_train', 'y_train_ctl', 'y_train_dys', 'y_train_exc', 'y_train_tide', 
         'x_test', 'y_test_ctl', 'y_test_dys', 'y_test_exc', 'y_test_tide']

j = 0
for i in tr_te:
    globals()['{}'.format(i)] = All[j]
    j += 1



#### Machine learning modeling #### 


# Random Forest Classifier (CTL)
# Target values [CTL level (TRUE = high, FALSE = low)]

print("## Random Forest Classifier ##")
print(" ")

All_RFC = RFC(x_train, x_test, y_train_ctl, y_test_ctl, model_save_folder)

best_model_CTL = All_RFC[0]
best_model_CTL_pred = All_RFC[1]


# Random Forest Classifier (Dys, Exc, TIDE)
print("## Random Forest Regressor ##")
print(" ")

target_list = ['tide', 'dys', 'exc']

for j in target_list:
    All_RFR = RFR(x_train, x_test, globals()["y_train_{}".format(j)], globals()["y_test_{}".format(j)], model_save_folder)
    globals()["best_model_{}".format(j)], globals()["best_model_{}_pred".format(j)] = All_RFR[0], All_RFR[1]


# Stepwise model 
print("Estimate performance Stepwise model")
stepwise_model(best_model_CTL_pred, best_model_dys_pred, best_model_exc_pred, y_test_dys, y_test_exc)



#### Calculate SHAP ####

model_list = ['CTL','dys','exc']

for i in model_list:
    calculate_shap(globals()["best_model_{}".format(i)], x_train, model_save_folder, i)


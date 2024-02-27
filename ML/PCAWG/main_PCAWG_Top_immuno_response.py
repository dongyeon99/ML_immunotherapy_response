
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

from utils_PCAWG_Top_immuno_response import *
from models_PCAWG_Top_immuno_response import *


#### Data prepare ####

# Path
data_folder = "/home/dong/python_work/TIDE/py_code/PCAWG/data/"
model_save_folder = "/ML/PCAWG/Top/model/"

# Data load
Top = prepare_tr_te(data_folder)

# Data split
tr_te = ['top1_x_train_ctl', 'top1_x_train_dys', 'top1_x_train_exc', 'top1_x_train_tide',
         'top2_x_train_ctl', 'top2_x_train_dys', 'top2_x_train_exc', 'top2_x_train_tide',
         'y_train_ctl', 'y_train_dys', 'y_train_exc', 'y_train_tide', 
         'top1_x_test_ctl', 'top1_x_test_dys', 'top1_x_test_exc', 'top1_x_test_tide',
         'top2_x_test_ctl', 'top2_x_test_dys', 'top2_x_test_exc', 'top2_x_test_tide',
         'y_test_ctl', 'y_test_dys', 'y_test_exc', 'y_test_tide']

j = 0
for i in tr_te:
    globals()['{}'.format(i)] = Top[j]
    j += 1 


#### Machine learning modeling #### 

Top_list = ['top1', 'top2']
target_list = ['tide', 'dys', 'exc']

#### LogisticRegression Classifier ####
# Target values [CTL level (TRUE = high, FALSE = low)]

print(" ")
print("## LogisticRegression Classifier ##")
print(" ")

for i in Top_list:
    
    Top_Logi = Logi(globals()["{}_x_train_ctl".format(i)], globals()["{}_x_test_ctl".format(i)], y_train_ctl, y_test_ctl, model_save_folder)

    globals()["{}_best_model_ctl".format(i)] = Top_Logi[0]
    globals()["{}_best_model_ctl_pred".format(i)] = Top_Logi[1]


#### linear Regression ####
print(" ")
print("## linear Regression ##")
print(" ")

for i in Top_list:
    for j in target_list:
        print("{}_MSE_{}_test:".format(i,j))
        Top_LR = LR(globals()["{}_x_train_{}".format(i, j)], globals()["{}_x_test_{}".format(i, j)], globals()["y_train_{}".format(j)], globals()["y_test_{}".format(j)], model_save_folder)
        
        globals()["{}_best_model_{}".format(i,j)] = Top_LR[0]
        globals()["{}_best_model_{}_pred".format(i,j)] = Top_LR[1]


#### Stepwise model ####
for i in Top_list:
    print("Estimate performance Stepwise model ({})".format(i))
    globals()["{}_MSE_stepwise_model".format(i)] = stepwise_model(globals()["{}_best_model_ctl_pred".format(i)], globals()["{}_best_model_dys_pred".format(i)], globals()["{}_best_model_exc_pred".format(i)], y_test_dys, y_test_exc)




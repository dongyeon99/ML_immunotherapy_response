
#### Package load ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from utils_All_Each_tumor_immuno_response import *
from models_All_Each_tumor_immuno_response import *


#### TCGA Each tumor type Data processing ####

TCGA_Each_folder = "/home/dong/python_work/TIDE/TCGA_each_tumor/"

TCGA_list = ["SKCM","UVM","LUAD","LUSC","BLCA","BRCA","CESC","COAD","ESCA","HNSC",
            "KIRC","KIRP","LGG","LIHC","OV","PAAD","SARC","STAD","UCEC"]

tr_te = ['x_train', 'y_train_ctl', 'y_train_dys', 'y_train_exc', 'y_train_tide', 
         'x_test', 'y_test_ctl', 'y_test_dys', 'y_test_exc', 'y_test_tide']

for i in TCGA_list:
    each = prepare_tr_te_each(i, TCGA_Each_folder)
    n = 0
    for j in tr_te:
        globals()['{}_{}'.format(i, j)] = each[n]
        n += 1


#### Machine learning modeling #### 

target_list = ['tide', 'dys', 'exc']

# Random Forest Classifier (CTL)
print("## Random Forest Classifier ##")
print(" ")

for i in TCGA_list:
    TCGA_RFC = RFC(i, globals()["{}_x_train".format(i)], globals()["{}_x_test".format(i)], globals()["{}_y_train_ctl".format(i)], globals()["{}_y_test_ctl".format(i)], TCGA_Each_folder)
    globals()["CTL_best_model_{}".format(i)], globals()["CTL_best_model_{}_pred".format(i,j)] = TCGA_RFC[0], TCGA_RFC[1]


# Random Forest Regressor (Dys, Exc, TIDE)
print("## Random Forest Regressor ##")
print(" ")

for i in TCGA_list:
    for j in target_list:
        TCGA_RFR = RFR(i, globals()["{}_x_train".format(i)], globals()["{}_x_test".format(i)], globals()["{}_y_train_{}".format(i,j)], globals()["{}_y_test_{}".format(i,j)], TCGA_Each_folder)
        globals()["best_model_{}_{}".format(i,j)], globals()["best_model_{}_{}_pred".format(i,j)] = TCGA_RFR[0], TCGA_RFR[1]


# Stepwise model 

for i in TCGA_list:
    print("Estimate performance Stepwise model ({})".format(i))
    stepwise_model(globals()["CTL_best_model_{}".format(i)], globals()["best_model_{}_dys_pred".format(i)], globals()["best_model_{}_exc_pred".format(i)], globals()["{}_y_test_dys".format(i)], globals()["{}_y_test_exc".format(i)])
    print("---------------------------------------------------")






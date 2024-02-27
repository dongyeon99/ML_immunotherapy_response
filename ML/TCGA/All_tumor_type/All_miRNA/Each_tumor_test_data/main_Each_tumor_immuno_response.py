
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

from utils_Each_tumor_immuno_response import *
from models_Each_tumor_immuno_response import *


#### TCGA Each tumor type Data processing ####

TCGA_Each_folder = "/ML_immunotherapy_response/Data/Result/TCGA/"

TCGA_list = ["SKCM","UVM","LUAD","LUSC","BLCA","BRCA","CESC","COAD","ESCA","HNSC",
            "KIRC","KIRP","LGG","LIHC","OV","PAAD","SARC","STAD","UCEC"]

tr_te = ['x_train', 'y_train_ctl', 'y_train_dys', 'y_train_exc', 'y_train_tide', 
         'x_test', 'y_test_ctl', 'y_test_dys', 'y_test_exc', 'y_test_tide']


# test train data split
for i in TCGA_list:
    each = prepare_tr_te_each(i, TCGA_Each_folder)
    n = 0
    for j in tr_te:
        globals()['{}_{}'.format(i, j)] = each[n]
        n += 1


# Each tumor type TCGA data merge 
for j in tr_te:
    globals()['list_data_{}'.format(j)] = []
    
for i in TCGA_list:
    for j in tr_te:
        tmp = globals()['{}_{}'.format(i, j)]
        globals()['list_data_{}'.format(j)].append(tmp)
        globals()['data_{}'.format(j)] = pd.concat(globals()['list_data_{}'.format(j)], ignore_index=True)
        


#### Machine learning modeling #### 

target_list = ['tide', 'dys', 'exc']


# Random Forest Classifier (CTL)
print("## Random Forest Classifier ##")
print(" ")

for i in TCGA_list:
    # Train model
    CTL_best_model= RFC(data_x_train, data_y_train_ctl)

    # Predict
    globals()["CTL_best_model_{}_pred".format(i)] = CTL_best_model.predict(globals()["{}_x_test".format(i)])
    
    # Score 
    CTL_F1_test = f1_score(globals()["{}_y_test_ctl".format(i)], globals()["CTL_best_model_{}_pred".format(i)])
    CTL_balanced_AUC_test = balanced_accuracy_score(globals()["{}_y_test_ctl".format(i)], globals()["CTL_best_model_{}_pred".format(i)])
    CTL_Accuracy = accuracy_score(globals()["{}_y_test_ctl".format(i)], globals()["CTL_best_model_{}_pred".format(i)])

    print("## CTL Classification Prediction ({}) ##".format(i))
    print("CTL_F1_Score:", CTL_F1_test)
    print("CTL_Balanced_Accuracy:", CTL_balanced_AUC_test)
    print("CTL_Accuracy:", CTL_Accuracy)
    print("-------------------------------------")
    print(" ")
    


# Random Forest Regressor (Dys, Exc, TIDE)
print("## Random Forest Regressor ##")
print(" ")

for i in TCGA_list:
    for j in target_list:
        # Train model
        globals()["best_model_{}".format(j)] = RFR(data_x_train, globals()["data_y_train_{}".format(j)])
            
        # Prediction
        globals()["best_model_{}_{}_pred".format(i,j)] = globals()["best_model_{}".format(j)].predict(globals()["{}_x_test".format(i)])

        # Score
        globals()["{}_MSE_{}_test".format(i,j)] = mean_squared_error(globals()["{}_y_test_{}".format(i,j)], globals()["best_model_{}_{}_pred".format(i,j)])
        print("({}) MSE_{}_test:".format(i,j) , globals()["{}_MSE_{}_test".format(i,j)])  



# Stepwise model 
for i in TCGA_list:
    print("Estimate performance Stepwise model ({})".format(i))
    stepwise_model(globals()["CTL_best_model_{}_pred".format(i)], globals()["best_model_{}_dys_pred".format(i)], globals()["best_model_{}_exc_pred".format(i)], globals()["{}_y_test_dys".format(i)], globals()["{}_y_test_exc".format(i)])
    print("---------------------------------------------------")






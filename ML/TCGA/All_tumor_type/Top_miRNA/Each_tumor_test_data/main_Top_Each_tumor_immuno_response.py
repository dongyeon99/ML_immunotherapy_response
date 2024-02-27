
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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from utils_Top_Each_tumor_immuno_response import *
from models_Top_Each_tumor_immuno_response import *


#### TCGA Each tumor type Data processing ####

TCGA_Each_folder = "/ML_immunotherapy_response/Data/Result/TCGA/"

TCGA_list = ["SKCM","UVM","LUAD","LUSC","BLCA","BRCA","CESC","COAD","ESCA","HNSC",
            "KIRC","KIRP","LGG","LIHC","OV","PAAD","SARC","STAD","UCEC"]

tr_te = ['top1_x_train_ctl', 'top1_x_train_dys', 'top1_x_train_exc', 'top1_x_train_tide',
         'top2_x_train_ctl', 'top2_x_train_dys', 'top2_x_train_exc', 'top2_x_train_tide',
         'y_train_ctl', 'y_train_dys', 'y_train_exc', 'y_train_tide', 
         'top1_x_test_ctl', 'top1_x_test_dys', 'top1_x_test_exc', 'top1_x_test_tide',
         'top2_x_test_ctl', 'top2_x_test_dys', 'top2_x_test_exc', 'top2_x_test_tide',
         'y_test_ctl', 'y_test_dys', 'y_test_exc', 'y_test_tide']


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

Top_list = ['top1', 'top2']
target_list = ['tide', 'dys', 'exc']


# LogisticRegression Classifier (CTL)
print(" ")
print("## LogisticRegression Classifier ##")
print(" ")

for j in Top_list:
    for i in TCGA_list:
        # Train model
        CTL_best_model= Logi(globals()['data_{}_x_train_ctl'.format(j)], data_y_train_ctl)

        # Predict
        globals()["{}_CTL_best_model_{}_pred".format(j,i)] = CTL_best_model.predict(globals()["{}_{}_x_test_ctl".format(i,j)])
        
        # Score 
        CTL_F1_test = f1_score(globals()["{}_y_test_ctl".format(i)], globals()["{}_CTL_best_model_{}_pred".format(j,i)])
        CTL_balanced_AUC_test = balanced_accuracy_score(globals()["{}_y_test_ctl".format(i)], globals()["{}_CTL_best_model_{}_pred".format(j,i)])
        CTL_Accuracy = accuracy_score(globals()["{}_y_test_ctl".format(i)], globals()["{}_CTL_best_model_{}_pred".format(j,i)])

        print("## CTL Classification Prediction ({}) ({}) ##".format(i,j))
        print("CTL_F1_Score:", CTL_F1_test)
        print("CTL_Balanced_Accuracy:", CTL_balanced_AUC_test)
        print("CTL_Accuracy:", CTL_Accuracy)
        print("-------------------------------------")
        print(" ")
 



# linear Regression (Dys, Exc, TIDE)
print(" ")
print("## linear Regression ##")
print(" ")

for n in Top_list:
    for i in TCGA_list:
        for j in target_list:
            # Train model
            globals()["best_model_{}".format(j)] = LR(globals()['data_{}_x_train_{}'.format(n,j)], globals()["data_y_train_{}".format(j)])
                
            # Prediction
            globals()["{}_best_model_{}_{}_pred".format(n,i,j)] = globals()["best_model_{}".format(j)].predict(globals()["{}_{}_x_test_{}".format(i,n,j)])

            # Score
            globals()["{}_{}_MSE_{}_test".format(n,i,j)] = mean_squared_error(globals()["{}_y_test_{}".format(i,j)], globals()["{}_best_model_{}_{}_pred".format(n,i,j)])
            print("({}) ({}) MSE_{}_test:".format(n,i,j) , globals()["{}_{}_MSE_{}_test".format(n,i,j)])  




# Stepwise model 
for n in Top_list:
    for i in TCGA_list:
        print("({}) Estimate performance Stepwise model ({})".format(n,i))
        stepwise_model(globals()["{}_CTL_best_model_{}_pred".format(n,i)], globals()["{}_best_model_{}_dys_pred".format(n,i)], globals()["{}_best_model_{}_exc_pred".format(n,i)], globals()["{}_y_test_dys".format(i)], globals()["{}_y_test_exc".format(i)])
        print("---------------------------------------------------")
        print(" ")






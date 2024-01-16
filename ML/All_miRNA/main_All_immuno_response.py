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

from utils_All_immuno_response import *
from models_All_immuno_response import *


#### Data prepare ####
# path
data_folder = "/ML_immunotherapy_response/Data/Result/"
model_folder = "/ML_immunotherapy_response/ML/All_miRNA/models/"

# data laod
data = pd.read_csv(os.path.join(data_folder,"data.csv"))

# Data split
All = prepare_tr_te(data)

tr_te = ['x_train', 'y_train_ctl', 'y_train_dys', 'y_train_exc', 'y_train_tide', 
         'x_test', 'y_test_ctl', 'y_test_dys', 'y_test_exc', 'y_test_tide']

j = 0
for i in tr_te:
    globals()['{}'.format(i)] = All[j]
    j += 1


#### Machine learning modeling #### 

# Random Forest Classifier (CTL)
print("## Random Forest Classifier ##")
print(" ")

All_RFC = RFC(x_train, x_test, y_train_ctl, y_test_ctl, model_folder)

CTL_best_model = All_RFC[0]
CTL_best_model_pred = All_RFC[1]


# Random Forest Classifier (Dys, Exc, TIDE)
print("## Random Forest Regressor ##")
print(" ")

target_list = ['tide', 'dys', 'exc']

for j in target_list:
    All_RFR = RFR(x_train, x_test, globals()["y_train_{}".format(j)], globals()["y_test_{}".format(j)], model_folder)
    globals()["best_model_{}".format(j)], globals()["best_model_{}_pred".format(j)] = All_RFR[0], All_RFR[1]


# Stepwise model 
print("Estimate performance Stepwise model")
stepwise_model(CTL_best_model_pred, best_model_dys_pred, best_model_exc_pred, y_test_dys, y_test_exc)


#### validation test ####
print(" ")
print("#### Validation Test #####")
print(" ")

# Data load
val_data = pd.read_csv(os.path.join(data_folder,"validation_data.csv"))

# Data processing
x_val = val_data.iloc[:, 2:len(val_data.columns)-5]

y_val_ctl = val_data[['CTL.flag']]

y_val_dys = val_data[['Dysfunction']]

y_val_exc = val_data[['Exclusion']]


# Validation test model 

# Pre-trained model load
CTL_model = joblib.load(os.path.join(model_folder,"All_CTL_model.pkl"))
Dys_model = joblib.load(os.path.join(model_folder,"All_RFR_Dysfunction.pkl"))
Exc_model = joblib.load(os.path.join(model_folder,"All_RFR_Exclusion.pkl"))

# Prediction
CTL_val_pred = CTL_model.predict(x_val)
Dys_val_pred = Dys_model.predict(x_val)
Exc_val_pred = Exc_model.predict(x_val)

# Score
F1_CTL_val = f1_score(y_val_ctl, CTL_val_pred)
MSE_Dys_val = mean_squared_error(y_val_dys, Dys_val_pred)
MSE_Exc_val = mean_squared_error(y_val_exc, Exc_val_pred)

print("F1_Score_CTL(Validation):", F1_CTL_val)
print(" ")
print("MSE_Dysfunction(Validation):", MSE_Dys_val)
print(" ")
print("MSE_Exclusion(Validation):", MSE_Exc_val)
print(" ")


# Validation Stepwise model MSE
print("Estimate performance Stepwise model (Validation)")
MSE_stepwise_model_val = stepwise_model(CTL_val_pred, Dys_val_pred, Exc_val_pred, y_val_dys, y_val_exc)

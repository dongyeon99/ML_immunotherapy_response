
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



#### Machine learning Modeling ####

#### [LogisticRegression Classifier] ####
# Target values [CTL level (TRUE = high, FALSE = low)]

def Logi(x_train, y_train):
    warnings.filterwarnings("ignore")
    # Machine learning load
    i = len(x_train.columns)
    
    # Top 1 
    if   i == 3:
        j = "top1"
        LRC = LogisticRegression(class_weight=None, penalty='l2', solver='liblinear', n_jobs=-1)
        
    # Top 2
    elif i == 2:
        j = "top2"
        LRC = LogisticRegression(class_weight=None, penalty='l2', solver='liblinear', n_jobs=-1)
    
    # best parameter model
    best_model = LRC
    
    # Train model
    best_model.fit(x_train, y_train.values.ravel())
    
    return best_model





#### [Random Forest Regressor] ####
# Target values [TIDE Score, Dysfunction Score and Exclusion Score]

def RFR(x_train, y_train):     
    # Machine learning load
    i = y_train.columns[0]
    
    if   i == 'Dysfunction':
        rfr = RandomForestRegressor(max_depth=25, n_estimators=700, n_jobs=-1)
    
    elif i == 'Exclusion':
        rfr = RandomForestRegressor(max_depth=20, n_estimators=300, n_jobs=-1)
            
    elif i == 'TIDE':
        rfr = RandomForestRegressor(max_depth=40, n_estimators=300, n_jobs=-1)

    # best parameter model
    best_model = rfr
    
    # Train model
    best_model.fit(x_train, y_train.values.ravel())
    
    return best_model


#### [linear Regression] ####
# Target values [TIDE Score, Dysfunction Score and Exclusion Score]

def LR(x_train, y_train):
    i = y_train.columns[0]
    j = len(x_train.columns)
    
    if   i == 'Dysfunction':
        if j == 5:
            n = "top1"
        else:
            n = "top2"
    
    elif i == 'Exclusion':
        if j == 12:
            n = "top1"
        else:
            n = "top2"
            
    elif i == 'TIDE':
        if j == 6:
            n = "top1"
        else:
            n = "top2"
    
    # Machine learning load
    LR = LinearRegression(n_jobs=-1)
    LR.fit(x_train, y_train.values.ravel()) 
    
    return LR



#### [Stepwise model build] ####
# CTL.flag [TRUE(high group) = Dysfunction Score -> TIDE Score, FALSE(low group) = Exclusion Score -> TIDE Score]

def stepwise_model(CTL_best_model_pred, best_model_dys_pred, best_model_exc_pred, y_test_dys, y_test_exc):
    warnings.filterwarnings("ignore")

    # Construct dataframes for predictions and test values
    ctl_pred = pd.DataFrame(CTL_best_model_pred, columns=['CTL.flag'])
    dys_pred = pd.DataFrame(best_model_dys_pred, columns=['Dysfunction_pred'])
    exc_pred = pd.DataFrame(best_model_exc_pred, columns=['Exclusion_pred'])
    dys_test = pd.DataFrame(y_test_dys, columns=['Dysfunction'])
    exc_test = pd.DataFrame(y_test_exc, columns=['Exclusion'])

    # Reset indices
    dys_test.reset_index(drop=True, inplace=True)
    exc_test.reset_index(drop=True, inplace=True)

    # Merge prediction and test dataframes
    total_pred = pd.concat([ctl_pred, dys_pred, dys_test, exc_pred, exc_test], axis=1)

    # Extract predictions and test values based on CTL flag
    pred_dys = total_pred.loc[total_pred['CTL.flag'] == True, ['Dysfunction_pred']] 
    test_dys = total_pred.loc[total_pred['CTL.flag'] == True, ['Dysfunction']]

    pred_exc = total_pred.loc[total_pred['CTL.flag'] == False, ['Exclusion_pred']]
    test_exc = total_pred.loc[total_pred['CTL.flag'] == False, ['Exclusion']]

    # Combine predictions and test values into final TIDE Score dataframe
    if not pred_dys.empty:
        pred_tide = pred_dys.rename(columns={"Dysfunction_pred": "TIDE_pred"})
        test_tide = test_dys.rename(columns={"Dysfunction": "TIDE"})        
    elif not pred_exc.empty:
        pred_tide = pred_exc.rename(columns={"Exclusion_pred": "TIDE_pred"})
        test_tide = test_exc.rename(columns={"Exclusion": "TIDE"})        
    else:
        pred_tide = pd.concat([pred_dys.rename(columns={"Dysfunction_pred": "TIDE_pred"}), 
                               pred_exc.rename(columns={"Exclusion_pred": "TIDE_pred"})], axis=0)
        test_tide = pd.concat([test_dys.rename(columns={"Dysfunction": "TIDE"}), 
                               test_exc.rename(columns={"Exclusion": "TIDE"})], axis=0)

    # Evaluate Prediction Results
    MSE_stepwise_model = mean_squared_error(test_tide, pred_tide)
    print("MSE stepwise model:", MSE_stepwise_model)
    
    return MSE_stepwise_model
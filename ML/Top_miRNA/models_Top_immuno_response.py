
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


#### Machine learning Modeling ####

#### [LogisticRegression Classifier] ####
# Target values [CTL level (TRUE = high, FALSE = low)]

def Logi(x_train, x_test, y_train, y_test, data_folder):
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
    globals()["{}_best_model_ctl".format(j)] = best_model
    
    # Prediction 
    globals()["{}_best_model_ctl_pred".format(j)] = best_model.predict(x_test)

    # Score
    globals()["CTL_F1_test_{}".format(j)] = f1_score(y_test, globals()["{}_best_model_ctl_pred".format(j)])
    globals()["CTL_balanced_AUC_test_{}".format(j)] = balanced_accuracy_score(y_test, globals()["{}_best_model_ctl_pred".format(j)])

    print("## CTL Prediction Score ##")
    print("{}_CTL_F1_Score:".format(j), globals()["CTL_F1_test_{}".format(j)])
    print("{}_CTL_Balanced_Accuracy:".format(j), globals()["CTL_balanced_AUC_test_{}".format(j)])
    print("-------------------------------------")
    print(" ")
    
    # Trained model save
    joblib.dump(best_model, os.path.join(data_folder,"{}_Logi_CTL.pkl".format(j)))
    
    return globals()["{}_best_model_ctl".format(j)], globals()["{}_best_model_ctl_pred".format(j)]


#### [linear Regression] ####
# Target values [TIDE Score, Dysfunction Score and Exclusion Score]

def LR(x_train, x_test, y_train, y_test, data_folder):
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
    
    # Model prediction
    LR_pred = LR.predict(x_test)
    
    # Score
    MSE_LR_test = mean_squared_error(y_test, LR_pred)
    print("MSE_Linear_test:", MSE_LR_test)
    print(" ") 

    # Trained model save
    joblib.dump(LR, os.path.join(data_folder,"{}_LR_{}.pkl".format(n, i)))
    
    return LR, LR_pred


#### [Stepwise model build] ####
# CTL.flag [TRUE(high group) = Dysfunction Score -> TIDE Score, FALSE(low group) = Exclusion Score -> TIDE Score]

def stepwise_model(CTL_best_model_pred, best_model_dys_pred, best_model_exc_pred, y_test_dys, y_test_exc):
    warnings.filterwarnings("ignore")
    
    ctl_pred = pd.DataFrame(CTL_best_model_pred, columns = ['CTL.flag'])

    dys_pred = pd.DataFrame(best_model_dys_pred, columns = ['Dysfunction_pred'])

    exc_pred = pd.DataFrame(best_model_exc_pred, columns = ['Exclusion_pred'])

    dys_test = pd.DataFrame(y_test_dys, columns = ['Dysfunction'])
    dys_test.reset_index(drop=True, inplace=True)

    exc_test = pd.DataFrame(y_test_exc, columns = ['Exclusion'])
    exc_test.reset_index(drop=True, inplace=True)

    total_pred = pd.concat([ctl_pred,
                                dys_pred, dys_test,
                                exc_pred, exc_test], axis=1)

    # CTL High -> Dysfunction Score
    total_pred_dys = total_pred[total_pred['CTL.flag'] == True]
    total_pred_dys

    pred_dys = total_pred_dys[['Dysfunction_pred']]

    test_dys = total_pred_dys[['Dysfunction']]


    # CTL Low -> Exclusion Score
    total_pred_exc = total_pred[total_pred['CTL.flag'] == False]
    total_pred_exc

    pred_exc = total_pred_exc[['Exclusion_pred']]

    test_exc = total_pred_exc[['Exclusion']]


    # Final TIDE Score
    # Prediction Values
    pred_tide_dys =  pred_dys
    pred_tide_dys.rename(columns={"Dysfunction_pred": "TIDE_pred"}, inplace=True)

    pred_tide_exc =  pred_exc
    pred_tide_exc.rename(columns={"Exclusion_pred": "TIDE_pred"}, inplace=True)

    pred_tide = pd.concat([pred_tide_dys, pred_tide_exc], axis=0)


    # Observed Values
    test_tide_dys =  test_dys
    test_tide_dys.rename(columns={"Dysfunction": "TIDE"}, inplace=True)

    test_tide_exc =  test_exc
    test_tide_exc.rename(columns={"Exclusion": "TIDE"}, inplace=True)

    test_tide = pd.concat([test_tide_dys, test_tide_exc], axis=0)


    #### Evaluate Prediction Results ####
    MSE_stepwise_model = mean_squared_error(test_tide, pred_tide)
    print("MSE stepwise model:" , MSE_stepwise_model)  
    
    return MSE_stepwise_model


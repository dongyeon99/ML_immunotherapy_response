
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

#### [Random Forest Classifier] ####
# Target values [CTL level (TRUE = high, FALSE = low)]

def RFC(x_train, x_test, y_train_ctl, y_test_ctl, data_folder):
    # model load (best parameter set)
    CTL_best_model = RandomForestClassifier(n_estimators = 250, criterion ='entropy', max_features = None, n_jobs=-1)
    
    # Train model
    CTL_best_model.fit(x_train, y_train_ctl.values.ravel())
    
    # Predict
    CTL_best_model_pred = CTL_best_model.predict(x_test)
    
    # Score 
    CTL_F1_test = f1_score(y_test_ctl, CTL_best_model_pred)
    CTL_balanced_AUC_test = balanced_accuracy_score(y_test_ctl, CTL_best_model_pred)
    
    print("## CTL Classification Prediction ##")
    print("CTL_F1_Score:", CTL_F1_test)
    print("CTL_Balanced_Accuracy:", CTL_balanced_AUC_test)
    print("-------------------------------------")
    print(" ")
    
    # Trained model save
    joblib.dump(CTL_best_model, os.path.join(data_folder,"All_CTL_model.pkl"))
    
    return CTL_best_model, CTL_best_model_pred


#### [Random Forest Regressor] ####
# Target values [TIDE Score, Dysfunction Score and Exclusion Score]

def RFR(x_train, x_test, y_train, y_test, data_folder):     
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
    globals()["best_model_{}".format(i)] = best_model
    
    # Prediction
    globals()["best_model_{}_pred".format(i)] = best_model.predict(x_test)
        
    # Score
    globals()["MSE_{}_test".format(i)] = mean_squared_error(y_test, globals()["best_model_{}_pred".format(i)])
    print("MSE_{}_test:".format(i) , globals()["MSE_{}_test".format(i)])  
    
    # Trained model save
    joblib.dump(best_model, os.path.join(data_folder,"All_RFR_{}.pkl".format(i)))
    
    return globals()["best_model_{}".format(i)], globals()["best_model_{}_pred".format(i)]


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


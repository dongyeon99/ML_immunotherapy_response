
#### package load ####
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


#### Machine learning Modeling ####

#### [LogisticRegression Classifier] ####
# Target values [CTL level (TRUE = high, FALSE = low)]

def Logi(x_train, x_test, y_train, y_test, model_save_folder):
    warnings.filterwarnings("ignore")
    # Machine learning load
    i = len(x_train.columns)
    
    # Top 1 
    if   i == 6:
        j = "top1"
        LRC = LogisticRegression(random_state=42, n_jobs=-1)
        
    # Top 2
    elif i == 4:
        j = "top2"
        LRC = LogisticRegression(random_state=42, n_jobs=-1)

    #set the parameters 
    parameters = {'class_weight':['balanced','dict',None],
                  'penalty':['l1','l2','elasticnet',None],
                  'solver':['lbfgs','newton-cg','newton-cholesky','sag','saga']}
    
    # grid search cv (10 fold cross validation)
    grid_search = GridSearchCV(LRC, parameters, cv=10, scoring='f1', n_jobs=-1)
    
    # Train the hyperparameters
    grid_search.fit(x_train, y_train.values.ravel())

    # best parameter
    print("LogisticRegression Classifier ({})".format(j))
    best_parameters = grid_search.best_estimator_.get_params()
    
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    print("")
    
    # best parameter model
    best_model = grid_search.best_estimator_
    
    # Train model
    best_model.fit(x_train, y_train.values.ravel())
    globals()["{}_best_model_ctl".format(j)] = best_model
    
    # Prediction 
    globals()["{}_best_model_ctl_pred".format(j)] = best_model.predict(x_test)

    # Score
    globals()["CTL_F1_test_{}".format(j)] = f1_score(y_test, globals()["{}_best_model_ctl_pred".format(j)])
    globals()["CTL_balanced_AUC_test_{}".format(j)] = balanced_accuracy_score(y_test, globals()["{}_best_model_ctl_pred".format(j)])
    globals()["CTL_AUC_test_{}".format(j)] = accuracy_score(y_test, globals()["{}_best_model_ctl_pred".format(j)])

    print("## CTL Prediction Score ##")
    print("{}_CTL_F1_Score:".format(j), globals()["CTL_F1_test_{}".format(j)])
    print("{}_CTL_Accuracy:".format(j), globals()["CTL_AUC_test_{}".format(j)] )
    print("{}_CTL_Balanced_Accuracy:".format(j), globals()["CTL_balanced_AUC_test_{}".format(j)])
    print("-------------------------------------")
    print(" ")
    
    # Trained model save
    joblib.dump(best_model, os.path.join(model_save_folder,"PCAWG_{}_Logi_CTL.pkl".format(j)))
    
    return globals()["{}_best_model_ctl".format(j)], globals()["{}_best_model_ctl_pred".format(j)]


#### [linear Regression] ####
# Target values [TIDE Score, Dysfunction Score and Exclusion Score]

def LR(x_train, x_test, y_train, y_test, model_save_folder):
    i = y_train.columns[0]
    j = len(x_train.columns)
    
    if   i == 'Dysfunction':
        if j == 10:
            n = "top1"
        else:
            n = "top2"
    
    elif i == 'Exclusion':
        if j == 22:
            n = "top1"
        else:
            n = "top2"
            
    elif i == 'TIDE':
        if j == 12:
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
    joblib.dump(LR, os.path.join(model_save_folder,"PCAWG_{}_LR_{}.pkl".format(n, i)))
    
    return LR, LR_pred


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

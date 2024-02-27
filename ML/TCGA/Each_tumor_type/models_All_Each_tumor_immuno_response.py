
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



#### Machine learning Modeling ####

#### [Random Forest Classifier] ####
# Target values [CTL level (TRUE = high, FALSE = low)]

def RFC(tumor_type, x_train, x_test, y_train, y_test, data_folder):
    warnings.filterwarnings("ignore")
    # model load 
    model = RandomForestClassifier(random_state=42)
    
    #set the parameters 
    parameters = {'criterion':['entropy', 'log_loss', 'gini'],
                  'max_features':['sqrt', 'log2', None],
                  'n_estimators':[100, 150, 200, 250, 300, 350, 400, 450]}
    
    # grid search cv (10 fold cross validation)
    grid_search = GridSearchCV(model, parameters, cv=10, scoring='f1', n_jobs=-1)
    
    # Train the hyperparameters
    grid_search.fit(x_train, y_train.values.ravel())

    # best parameter
    print("Random Forest Classifier ({})".format(tumor_type))
    best_parameters = grid_search.best_estimator_.get_params()
    
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    # Train the best RFR model
    CTL_best_model = grid_search.best_estimator_
    CTL_best_model.fit(x_train, y_train.values.ravel())
    
    # Predict
    CTL_best_model_pred = CTL_best_model.predict(x_test)
    
    # Score 
    CTL_F1_test = f1_score(y_test, CTL_best_model_pred)
    CTL_AUC_test = accuracy_score(y_test, CTL_best_model_pred)
    
    print("## CTL Classification Prediction ##")
    print("CTL_F1_Score ({}) :".format(tumor_type), CTL_F1_test)
    print("CTL_Accuracy ({}) :".format(tumor_type), CTL_AUC_test)
    print("-------------------------------------")
    print(" ")
    
    # Trained model save
    joblib.dump(CTL_best_model, os.path.join(data_folder,"All_RFC_{}_CTL.pkl".format(tumor_type)))
    
    return CTL_best_model, CTL_best_model_pred



#### [Random Forest Regressor] ####
# Target values [TIDE Score, Dysfunction Score and Exclusion Score]

def RFR(tumor_type, x_train, x_test, y_train, y_test, data_folder):
    warnings.filterwarnings("ignore")
    # Machine learning load
    i = y_train.columns[0]
    
    model = RandomForestRegressor(random_state=42)

    #set the parameters 
    parameters = {'max_depth':[15,20,25,30,35,40,45],
                  'n_estimators':[200,300,400,500,600,700,800,900]}

    # grid search cv (10 fold cross validation)
    mse = make_scorer(mean_squared_error,greater_is_better=False)
    grid_search = GridSearchCV(model, parameters, cv=10, scoring=mse, n_jobs=-1)

    # Train the hyperparameters
    grid_search.fit(x_train, y_train.values.ravel())

    # best parameter
    print("Random Forest Regression best parameter ({}) ({})".format(tumor_type, i))
    best_parameters = grid_search.best_estimator_.get_params()
    
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    # Train the best RFR model
    best_RFR = grid_search.best_estimator_
    best_RFR.fit(x_train, y_train.values.ravel())

    # Predict
    RFR_pred = best_RFR.predict(x_test)

    # Score
    RFR_mse = mean_squared_error(y_test, RFR_pred)

    print('best RFR MSE ({}) ({}):'.format(tumor_type, i), RFR_mse)
    print("-------------------------------------")
    print(" ")
    
    # Trained model save
    joblib.dump(best_RFR, os.path.join(data_folder,"All_RFR_{}_{}.pkl".format(tumor_type, i)))
    
    return best_RFR, RFR_pred



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

# -*- coding: utf-8 -*-
"""
@author: dong
"""

########################
## package load ##

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score






#### Data load ####

data = pd.read_csv("/Data/Result/TIDE_TCGA19_miR.csv")


### Data preprocessing ###

### Train and Test Data Split ###
train, test = train_test_split(data, train_size=0.8, stratify = data['CTL.flag'], random_state=42)


### Train dataset ###
x_train = train.iloc[:,2:1883]
#print(x_train.head())

y_train_ctl = train[['CTL.flag']]
#print(y_train_ctl)

y_train_dys = train[['Dysfunction']]
#print(y_train_dys)

y_train_exc = train[['Exclusion']]
#print(y_train_exc)

y_train_tide = train[['TIDE']]
#print(y_train_tide)


### Test dataset ###
x_test = test.iloc[:,2:1883]
#print(x_test.head())

y_test_ctl = test[['CTL.flag']]
#print(y_test_ctl)

y_test_dys = test[['Dysfunction']]
#print(y_test_dys)

y_test_exc = test[['Exclusion']]
#print(y_test_exc)

y_test_tide = test[['TIDE']]
#print(y_test_exc)





#### Machine learning [Random Forest Classifier] ####

# Target values [CTL level (TRUE = high, FALSE = low)]

print("## Random Forest Classifier ##")
print(" ")

# Classification Model load
RFC = RandomForestClassifier(n_estimators = 250, criterion ='entropy', max_features = None, n_jobs=-1)

# best parameter model prediction
CTL_best_model = RFC

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





#### Machine learning [Random Forest Regressor] ####

# Target values [TIDE Score, Dysfunction Score and Exclusion Score]

print("## Random Forest Regressor ##")
print(" ")

TIDE_list = ['tide', 'dys', 'exc']

for i in TIDE_list:
    
    # X and Y data load
    X_train = x_train
    X_test = x_test
    Y_train = globals()["y_train_{}".format(i)]
    Y_test = globals()["y_test_{}".format(i)]
    
    # Machine learning load
    if   i == 'dys':
        rfr = RandomForestRegressor(max_depth=25, n_estimators=700, n_jobs=-1)
    
    elif i == 'exc':
        rfr = RandomForestRegressor(max_depth=20, n_estimators=300, n_jobs=-1)
            
    elif i == 'tide':
        rfr = RandomForestRegressor(max_depth=40, n_estimators=300, n_jobs=-1)

    # best parameter model
    best_model = rfr
    
    # Train model
    best_model.fit(X_train, Y_train.values.ravel())
    globals()["best_model_{}".format(i)] = best_model
    
    # Prediction
    globals()["best_model_{}_pred".format(i)] = best_model.predict(X_test)
        
    # Score
    globals()["MSE_{}_test".format(i)] = mean_squared_error(Y_test, globals()["best_model_{}_pred".format(i)])
    print("MSE_{}_test:".format(i) , globals()["MSE_{}_test".format(i)])  






#### Multi model build ####

#### Prediction Result merge ####
# CTL.flag [TRUE(high group) = Dysfunction Score -> TIDE Score, FALSE(low group) = Exclusion Score -> TIDE Score]


# Prediction Results split

ctl_pred_test = pd.DataFrame(CTL_best_model_pred, columns = ['CTL.flag'])

dys_pred_test = pd.DataFrame(best_model_dys_pred, columns = ['Dysfunction_pred'])

exc_pred_test = pd.DataFrame(best_model_exc_pred, columns = ['Exclusion_pred'])

dys_test = pd.DataFrame(y_test_dys, columns = ['Dysfunction'])
dys_test.reset_index(drop=True, inplace=True)

exc_test = pd.DataFrame(y_test_exc, columns = ['Exclusion'])
exc_test.reset_index(drop=True, inplace=True)

total_pred_test = pd.concat([ctl_pred_test,
                            dys_pred_test,dys_test,
                            exc_pred_test,exc_test], axis=1)

# CTL High -> Dysfunction Score
total_pred_test_dys = total_pred_test[total_pred_test['CTL.flag'] == True]
total_pred_test_dys

pred_dys = total_pred_test_dys[['Dysfunction_pred']]

test_dys = total_pred_test_dys[['Dysfunction']]


# CTL Low -> Exclusion Score
total_pred_test_exc = total_pred_test[total_pred_test['CTL.flag'] == False]
total_pred_test_exc

pred_exc = total_pred_test_exc[['Exclusion_pred']]

test_exc = total_pred_test_exc[['Exclusion']]


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
print("## Evaluate Prediction Results ##")

total_MSE_tide_test = mean_squared_error(test_tide, pred_tide)
print("Multi_MSE_TIDE:" , total_MSE_tide_test)  







#########################################################
############### Validation Test #########################
#########################################################

print(" ")
print("#################################################")
print("############### Validation Test #################")
print("#################################################")
print(" ")


# Data load 
val_data = pd.read_csv("/Data/Result/TIDE_TCGA_val_miR.csv")

# Data split 
x_val = val_data.iloc[:,2:1883]

y_val_ctl = val_data[['CTL.flag']]

y_val_dys = val_data[['Dysfunction']]

y_val_exc = val_data[['Exclusion']]



#### Validation test Classification model [CTL] ####

# Prediction
RFC_val_pred = CTL_best_model.predict(x_val)

# Score
F1_CTL_val = f1_score(y_val_ctl, RFC_val_pred)
Balanced_AUC_CTL_val = balanced_accuracy_score(y_val_ctl, RFC_val_pred)

print("F1_Score_CTL:", F1_CTL_val)
print(" ")


#### Validation test Regressor model [Dysfunction] ####

# Predict
RFR_dys_val_pred = best_model_dys.predict(x_val)
RFR_dys_val_pred

# score
mse_RFR_dys_val = mean_squared_error(y_val_dys, RFR_dys_val_pred)
print("Validation_MSE_Dysfunction:", mse_RFR_dys_val)
print(" ")


#### Validation test Regressor model [Exclusion] ####

# Predict
RFR_exc_val_pred = best_model_exc.predict(x_val)
RFR_exc_val_pred

# score
mse_RFR_exc_val = mean_squared_error(y_val_exc, RFR_exc_val_pred)
print("Validation_MSE_Exclusion:", mse_RFR_exc_val)
print(" ")


#### Validation test Multi model ####

#### Prediction Result merge ####

# Prediction Results split

ctl_pred_val = pd.DataFrame(RFC_val_pred, columns = ['CTL.flag'])

dys_pred_val = pd.DataFrame(RFR_dys_val_pred, columns = ['Dysfunction_pred'])

exc_pred_val = pd.DataFrame(RFR_exc_val_pred, columns = ['Exclusion_pred'])

dys_val = pd.DataFrame(y_val_dys, columns = ['Dysfunction'])
dys_val.reset_index(drop=True, inplace=True)

exc_val = pd.DataFrame(y_val_exc, columns = ['Exclusion'])
exc_val.reset_index(drop=True, inplace=True)

total_pred_val = pd.concat([ctl_pred_val,
                            dys_pred_val, dys_val,
                            exc_pred_val, exc_val], axis=1)


# CTL High -> Dysfunction Score
total_pred_val_dys = total_pred_val[total_pred_val['CTL.flag'] == True]

pred_dys_val = total_pred_val_dys[['Dysfunction_pred']]

val_dys = total_pred_val_dys[['Dysfunction']]


# CTL Low -> Exclusion Score
total_pred_val_exc = total_pred_val[total_pred_val['CTL.flag'] == False]

pred_exc_val = total_pred_val_exc[['Exclusion_pred']]

val_exc = total_pred_val_exc[['Exclusion']]


# Final TIDE Score

# Prediction Values
pred_tide_val_dys =  pred_dys_val
pred_tide_val_dys.rename(columns={"Dysfunction_pred": "TIDE_pred"}, inplace=True)

pred_tide_val_exc =  pred_exc_val
pred_tide_val_exc.rename(columns={"Exclusion_pred": "TIDE_pred"}, inplace=True)

pred_tide_val = pd.concat([pred_tide_val_dys, pred_tide_val_exc], axis=0)



# Observed Values
test_tide_val_dys =  val_dys
test_tide_val_dys.rename(columns={"Dysfunction": "TIDE"}, inplace=True)

test_tide_val_exc =  val_exc
test_tide_val_exc.rename(columns={"Exclusion": "TIDE"}, inplace=True)

val_tide = pd.concat([test_tide_val_dys, test_tide_val_exc], axis=0)


# Estimate performance Multi model (TIDE)
val_multi_MSE_tide_test = mean_squared_error(val_tide, pred_tide_val)
print("Validation_Multi_MSE_TIDE:" , val_multi_MSE_tide_test)  
print(" ")




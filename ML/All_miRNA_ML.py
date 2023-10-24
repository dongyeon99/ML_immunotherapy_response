# -*- coding: utf-8 -*-
"""
@author: dong
"""

########################
## package load ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from yellowbrick.classifier import ROCAUC

import shap



#### Data load ####

data = pd.read_csv("/home/dong/python_work/TIDE/TIDE_Score/TIDE_Score_TCGA19_miR_rm_dup.csv")
#data.head()

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

##### Cross validation grid search #########

# Hyperparameter tuning and Estimate performance

# Target values [CTL level (TRUE = high, FALSE = low)]


# Classification Model load
RFC = RandomForestClassifier()

# Set parameters
RFC_parameters = {'n_estimators':[100,150,200,250,300,350,400,450],
                    'criterion':['gini','entropy','log_loss'],
                    'max_features':['sqrt','log2',None]}


# Grid Search CV (CV = StratifiedKFold)
grid_search_rfc = GridSearchCV(RFC, RFC_parameters, cv=10, scoring='f1', n_jobs=60)

# Train the hyperparameters
grid_search_rfc.fit(x_train, y_train_ctl.values.ravel())

# Grid Search CV results Save
scores_rfc = pd.DataFrame(grid_search_rfc.cv_results_)
scores_rfc.to_csv('/home/dong/python_work/TIDE/final_TIDE_Score/hyperparameter/hyperparameter_All_CTL_classifier.csv')

# best parameter
print("## Random Forest Classifier ##")
print("## Grid Search CV Results ##")
print("CTL_Random_Forest_Classification_Best_Parameter:", grid_search_rfc.best_estimator_)
#print("CTL_Random_Forest_Classification_F1_Score:", grid_search_rfc.best_score_)
print("-------------------------------------")

# best parameter model prediction
CTL_best_model = grid_search_rfc.best_estimator_
CTL_best_model_pred = CTL_best_model.predict(x_test)

# Score
CTL_F1_test = f1_score(y_test_ctl, CTL_best_model_pred)
CTL_balanced_AUC_test = balanced_accuracy_score(y_test_ctl, CTL_best_model_pred)

print("## CTL Classification Prediction ##")
print("CTL_F1_Score:", CTL_F1_test)
print("CTL_Balanced_Accuracy:", CTL_balanced_AUC_test)
print("-------------------------------------")


# ROCAUC Plot
visualizer = ROCAUC(CTL_best_model)
visualizer.fit(x_train, y_train_ctl.values.ravel())
visualizer.score(x_test, y_test_ctl)
visualizer.show(outpath="/home/dong/python_work/TIDE/final_TIDE_Score/Figure/ROC_Curves_All_CTL_Classifier.png")

# Clear reset plot
plt.clf() 




#### Machine learning [Random Forest Regressor] ####

# Target values [TIDE Score (Dysfunction Score or Exclusion Score)]

# CTL.flag [TRUE(high group) = Dysfunction Score, FALSE(low group) = Exclusion Score]

# Cross validation grid search 

# Hyperparameter tuning and Estimate performance

# Target values [TIDE Score (Dysfunction Score or Exclusion Score)]

print("## Random Forest Regressor ##")

TIDE_list = ['tide', 'dys', 'exc']

for i in TIDE_list:
    # X and Y data load
    X_train = x_train
    X_test = x_test
    Y_train = globals()["y_{}_train".format(i)]
    Y_test = globals()["y_{}_test".format(i)]
    
    # Machine learning load
    rfr = RandomForestRegressor()

    #set the parameters 
    rf_parameters = {'max_depth':[15,20,25,30,35,40,45], 'n_estimators':[200,300,400,500,600,700,800,900]}

    # grid search cv
    grid_search_rf = GridSearchCV(rfr, rf_parameters, cv=10, scoring='neg_mean_squared_error',
                                       n_jobs=60)

    # Train the hyperparameters
    grid_search_rf.fit(X_train, Y_train.values.ravel())

    # Grid search cv results save
    scores_rf = pd.DataFrame(grid_search_rf.cv_results_)
    scores_rf.to_csv('/home/dong/python_work/TIDE/final_TIDE_Score/hyperparameter/hyperparameter_Score_All_{}.csv'.format(i))

    # best parameter
    print("{}_Best_Parameter:".format(i), grid_search_rf.best_estimator_)
    #print("NMSE_{}:".format(i), grid_search_rf.best_score_)

    # best parameter model prediction
    best_model = grid_search_rf.best_estimator_
    globals()["best_model_{}".format(i)] = best_model
    globals()["best_model_{}_pred".format(i)] = best_model.predict(X_test)
        
    # Score
    globals()["MSE_{}_test".format(i)] = mean_squared_error(Y_test, globals()["best_model_{}_pred".format(i)])
    print("MSE_{}_Score_test:".format(i) , globals()["MSE_{}_test".format(i)])  




# Scatter Plot & Correlation

for i in TIDE_list:
    
    if i == 'dys':
        j = "Dysfunction"
    
    elif i == 'exc':
        j = "Exclusion"
        
    #elif i == 'tide':
    #    j = "TIDE"
    
    # data set
    obse = globals()["best_model_{}".format(i)]
    pred = globals()["best_model_{}_pred".format(i)]
    
    obse.columns=[j]
    pred.columns=[j]
    
    print("{} Correlation:".format(j), obse[j].corr(pred[j]))
    
    # Scatter plot
    plt.scatter(obse[j], pred[j], s=3)
    
    #regression line
    m, b = np.polyfit(obse[j], pred[j], 1)
    plt.plot(obse[j], m*obse[j]+b, color='red')

    # plot design
    plt.xlabel('Observed {}'.format(j) , fontsize = 14)
    plt.ylabel('Predicted {}'.format(j), fontsize = 14)

    # save plot
    plt.savefig('/home/dong/python_work/TIDE/Figure/Scatter plot of TCGA19 {}.png'.format(j), dpi=1200)
    
    # Clear reset plot
    plt.clf() 





#### Prediction Result merge ####

# CTL.flag [TRUE(high group) = Dysfunction Score, FALSE(low group) = Exclusion Score]

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


#############################################
#  Evaluate Prediction Results

print("## Evaluate Prediction Results ##")

total_MSE_tide_test = mean_squared_error(test_tide, pred_tide)
print("Multi_MSE_TIDE:" , total_MSE_tide_test)  




### Scatter Plot & Correlation

# Regression Line [correlation]

print("Multi TIDE Correlation:", test_tide['TIDE'].corr(pred_tide['TIDE_pred']))

# Scatter plot for test data
plt.scatter(test_tide['TIDE'], pred_tide['TIDE_pred'], s=3)

#regression line
m, b = np.polyfit(test_tide['TIDE'], pred_tide['TIDE_pred'], 1)
plt.plot(test_tide['TIDE'], m*test_tide['TIDE']+b, color='red')

# plot design
plt.xlabel('Observed TIDE', fontsize = 14)
plt.ylabel('Predicted TIDE', fontsize = 14)

# save plot
plt.savefig('/home/dong/python_work/TIDE/Figure/Scatter plot of All Multi TIDE.png', dpi=1200)

# Clear reset plot
plt.clf() 








############### Validation Test ################

print("############### Validation Test ################")

#### Data load ####

val_data = pd.read_csv("/home/dong/python_work/TIDE/TIDE_Score/validation/TIDE_Score_TCGA_Val_miR_rm_dup.csv")


# Data split 
x_val = val_data.iloc[:,2:1883]

y_val_ctl = val_data[['CTL.flag']]

y_val_dys = val_data[['Dysfunction']]

y_val_exc = val_data[['Exclusion']]



# Validation test Classification model [CTL]

# Prediction
RFC_val_pred = RFC.predict(x_val)

# Score
F1_CTL_val = f1_score(y_val_ctl, RFC_val_pred)
Balanced_AUC_CTL_val = balanced_accuracy_score(y_val_ctl, RFC_val_pred)

print("F1_Score_CTL:", F1_CTL_val)
print("Balanced_AUC_CTL:", Balanced_AUC_CTL_val)


# ROCAUC Plot
visualizer_val = ROCAUC(RFC, title= my_title)

visualizer_val.fit(x_train, y_train_ctl.values.ravel())
visualizer_val.score(x_val, y_val_ctl.values.ravel())

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)

visualizer_val.show(outpath="/home/dong/python_work/TIDE/Figure/TCGA19/ROC_Curves_TCGA_val_CTL_Classifier.png", dpi=1200)

# Clear reset plot
plt.clf() 



#### Validation test Regressor model [Dysfunction]

# Predict
RFR_dys_val_pred = best_model_dys.predict(x_val)
RFR_dys_val_pred

# score
mse_RFR_dys_val = mean_squared_error(y_val_dys, RFR_dys_val_pred)
print("MSE_Dysfunction:", mse_RFR_dys_val)


# Dysfunction Scatter Plot
# Observed purity
TCGA_val_dys_obse = y_val_dys

# Predicted purity
TCGA_val_dys_pred = pd.DataFrame(RFR_dys_val_pred)
TCGA_val_dys_pred.columns=["Dysfunction"]

print("Validation Dysfunction Correlation:", TCGA_val_dys_obse['Dysfunction'].corr(TCGA_val_dys_pred['Dysfunction']))

# Scatter plot for test data
plt.scatter(TCGA_val_dys_obse['Dysfunction'], TCGA_val_dys_pred['Dysfunction'], s=3)

#regression line
m, b = np.polyfit(TCGA_val_dys_obse['Dysfunction'], TCGA_val_dys_pred['Dysfunction'], 1)
plt.plot(TCGA_val_dys_obse['Dysfunction'], m*TCGA_val_dys_obse['Dysfunction']+b, color='red')

#plt.plot(TCGA_dys_obse, TCGA_dys_pred, color='red')

# plot design
plt.xlabel('Observed Dysfunction', fontsize=14)
plt.ylabel('Predicted Dysfunction', fontsize=14)

# save plot
plt.savefig('/home/dong/python_work/TIDE/Figure/TCGA19/Scatter plot of TCGA val Dysfunction.png',dpi=1200)

# Clear reset plot
plt.clf() 



#### Validation test Regressor model [Exclusion]

# Predict
RFR_exc_val_pred = best_model_exc.predict(x_val)
RFR_exc_val_pred

# score
mse_RFR_exc_val = mean_squared_error(y_val_exc, RFR_exc_val_pred)
print("MSE_Exclusion:", mse_RFR_exc_val)


# Exclusion Scatter Plot
# Observed purity
TCGA_val_Exc_obse = y_val_exc

# Predicted purity
TCGA_val_Exc_pred = pd.DataFrame(RFR_exc_val_pred)
TCGA_val_Exc_pred.columns=["Exclusion"]

print("Validation Exclusion Correlation:", TCGA_val_Exc_obse['Exclusion'].corr(TCGA_val_Exc_pred['Exclusion']))

# Scatter plot for test data
plt.scatter(TCGA_val_Exc_obse['Exclusion'], TCGA_val_Exc_pred['Exclusion'], s=3)

#regression line
m, b = np.polyfit(TCGA_val_Exc_obse['Exclusion'], TCGA_val_Exc_pred['Exclusion'], 1)
plt.plot(TCGA_val_Exc_obse['Exclusion'], m*TCGA_val_Exc_obse['Exclusion']+b, color='red')

# plot design
plt.xlabel('Observed Exclusion', fontsize=14)
plt.ylabel('Predicted Exclusion', fontsize=14)

# save plot
plt.savefig('/home/dong/python_work/TIDE/Figure/TCGA19/scatter plot of TCGA val Exclusion.png', dpi=1200)

# Clear reset plot
plt.clf() 



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


# Multi model MSE (TIDE)
val_multi_MSE_tide_test = mean_squared_error(val_tide, pred_tide_val)
print("Val_Multi_MSE_TIDE:" , val_multi_MSE_tide_test)  


########### Validation data Scatter plot #############


for i in TIDE_list:
    
    #if i == 'dys':
    #    j = "Dysfunction"
    
    #elif i == 'exc':
    #    j = "Exclusion"
        
    if i == 'tide':
        j = "TIDE"
    
    # data set
    obse = globals()["val_{}".format(i)]
    pred = globals()["pred_{}_val".format(i)]
    
    obse.columns=[j]
    pred.columns=[j]
    
    print("Val Multi {} Correlation:".format(j), obse[j].corr(pred[j]))
    
    # Scatter plot
    plt.scatter(obse[j], pred[j], s=3)
    
    #regression line
    m, b = np.polyfit(obse[j], pred[j], 1)
    plt.plot(obse[j], m*obse[j]+b, color='red')

    # plot design
    plt.xlabel('Observed {}'.format(j) , fontsize = 14)
    plt.ylabel('Predicted {}'.format(j), fontsize = 14)

    # save plot
    plt.savefig('/home/dong/python_work/TIDE/Figure/Scatter plot of TCGA19 Multi Val {}.png'.format(j), dpi=1200)










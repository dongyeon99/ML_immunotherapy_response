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
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler


from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

from yellowbrick.classifier import ROCAUC

import shap
from auto_shap.auto_shap import generate_shap_values
from auto_shap.auto_shap import produce_shap_values_and_summary_plots



###################
#### Data load ####
###################


data = pd.read_csv("/home/dong/python_work/TIDE/TIDE_Score/TIDE_Score_TCGA19_miR_rm_dup.csv")

###########################
#### Data preprocessing ###
###########################


### Train and Test Data Split ###

train, test = train_test_split(data, train_size=0.8, stratify = data['CTL.flag'], random_state=42)

x_train = train.iloc[:,2:1883]

x_test = test.iloc[:,2:1883]

### Standard Scaling ###

scaler = StandardScaler()

x_list = list(x_train.columns)

x_train_sc = x_train.copy()
x_train_sc[x_list] = scaler.fit_transform(x_train[x_list])
#print(x_train_sc)

x_test_sc = x_test.copy()
x_test_sc[x_list] = scaler.transform(x_test[x_list])
#print(x_test_sc)



### Train dataset ###

print( "Top 1 ( mean(SHAP value) > 0.01 )")
print("CTL model: Top 3")
print("Dysfunction model: Top 5")
print("Exclusion model: Top 12")
print(" ")

top1_x_train_ctl = x_train_sc[['hsa.mir.155','hsa.mir.150','hsa.mir.4772']]

top1_x_train_tide = x_train_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183','hsa.mir.155','hsa.mir.345']]

top1_x_train_dys = x_train_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a','hsa.mir.210']]

top1_x_train_exc = x_train_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150','hsa.mir.192',
                               'hsa.mir.141','hsa.mir.155','hsa.mir.200c','hsa.mir.493','hsa.mir.708','hsa.mir.142']]




print( "Top 2 ( mean(SHAP value) > 0.02 )")
print("CTL model: Top 2")
print("Dysfunction model: Top 4")
print("Exclusion model: Top 5")
print(" ")

top2_x_train_ctl = x_train_sc[['hsa.mir.150','hsa.mir.155']]

top2_x_train_tide = x_train_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183']]

top2_x_train_dys = x_train_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a']]

top2_x_train_exc = x_train_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150']]


y_train_ctl = train[['CTL.flag']]
#print(y_train_ctl)

y_train_dys = train[['Dysfunction']]
#print(y_train_dys)

y_train_exc = train[['Exclusion']]
#print(y_train_exc)

y_train_tide = train[['TIDE']]
#print(y_train_tide)



### Test dataset ###

# Top 1  ( mean(SHAP value) > 0.01 )

top1_x_test_ctl = x_test_sc[['hsa.mir.155','hsa.mir.150','hsa.mir.4772']]

top1_x_test_tide = x_test_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183','hsa.mir.155','hsa.mir.345']]

top1_x_test_dys = x_test_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a','hsa.mir.210']]

top1_x_test_exc = x_test_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150','hsa.mir.192',
                               'hsa.mir.141','hsa.mir.155','hsa.mir.200c','hsa.mir.493','hsa.mir.708','hsa.mir.142']]


# Top 2  ( mean(SHAP value) > 0.02 )

top2_x_test_ctl = x_test_sc[['hsa.mir.155','hsa.mir.4772']]

top2_x_test_tide = x_test_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183']]

top2_x_test_dys = x_test_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a']]

top2_x_test_exc = x_test_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150']]



y_test_ctl = test[['CTL.flag']]
#print(y_test_ctl)

y_test_dys = test[['Dysfunction']]
#print(y_test_dys)

y_test_exc = test[['Exclusion']]
#print(y_test_exc)

y_test_tide = test[['TIDE']]
#print(y_test_exc)



############################################
####### Machine Learning Model ##########
###########################################

Top_list = ['top1', 'top2']
TIDE_list = ['tide', 'dys', 'exc']
total = ['ctl','dys','exc']

### 1. Random Forest Classifier ###

##### Cross validation grid search #########

# Hyperparameter tuning and Estimate performance

# Target values [CTL level (TRUE = high, FALSE = low)]

print(" ")
print("## 1. LogisticRegression Classifier ##")
print(" ")

#Top_list = ['top6', 'top4']
#TIDE_list = ['tide', 'dys', 'exc']
#total = ['ctl','tide','dys','exc']

for i in Top_list:
    # X and Y data load
    X_train = globals()["{}_x_train_ctl".format(i)]
    X_test  = globals()["{}_x_test_ctl".format(i)]
    Y_train = y_train_ctl
    Y_test  = y_test_ctl
    
    # Classification Model load
    LRC = linear_model.LogisticRegression()

    # Set parameters
    
    LRC_parameters = {'penalty':['l1', 'l2', 'elasticnet', None],
                      'class_weight': ['balanced', None],
                      'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}


    # Grid Search CV (CV = StratifiedKFold)
    grid_search_LRC = GridSearchCV(LRC, LRC_parameters, cv=10, scoring='f1', n_jobs=60)

    # Train the hyperparameters
    grid_search_LRC.fit(X_train, Y_train.values.ravel())

    # Grid Search CV results Save
    scores_LRC = pd.DataFrame(grid_search_LRC.cv_results_)
    scores_LRC.to_csv('/home/dong/python_work/TIDE/hyperparameter/SHAP_Top_miRNA/hyperparameter_{}_CTL_LogisticRegression_classifier.csv'.format(i))

    # best parameter
    print("## Grid Search CV Results ##")
    print("{}_CTL_LogisticRegression_classifier_Best_Parameter:".format(i), grid_search_LRC.best_estimator_)
    print("{}_CTL_LogisticRegression_classifier_F1_Score:".format(i), grid_search_LRC.best_score_)
    print("-------------------------------------")

    # best parameter model prediction
    best_model = grid_search_LRC.best_estimator_
    globals()["{}_best_model_ctl".format(i)] = best_model
    globals()["{}_best_model_ctl_pred".format(i)] = best_model.predict(X_test)

    # Score
    globals()["CTL_F1_test_{}".format(i)] = f1_score(Y_test, globals()["{}_best_model_ctl_pred".format(i)])
    globals()["CTL_balanced_AUC_test_{}".format(i)] = balanced_accuracy_score(Y_test, globals()["{}_best_model_ctl_pred".format(i)])

    print("## CTL Classification Prediction test Score ##")
    print("{}_CTL_F1_Score:".format(i), globals()["CTL_F1_test_{}".format(i)])
    print("{}_CTL_Balanced_Accuracy:".format(i), globals()["CTL_balanced_AUC_test_{}".format(i)])
    print("-------------------------------------")
    print(" ")
    

    # ROCAUC Plot
    globals()["visualizer_{}".format(i)] = ROCAUC(globals()["{}_best_model_ctl".format(i)])
    globals()["visualizer_{}".format(i)].fit(X_train, Y_train.values.ravel())
    globals()["visualizer_{}".format(i)].score(X_test, Y_test)
    globals()["visualizer_{}".format(i)].show(outpath="/home/dong/python_work/TIDE/Figure/SHAP_Top_miRNA/ROC_Curves_{}_CTL_LogisticRegression_Classifier.png".format(i), dpi=1200)

    # Clear reset plot
    plt.clf() 




####################################################################
########### 2. Random Forest Regressor ################

# Target values [TIDE Score (Dysfunction Score or Exclusion Score)]

# CTL.flag [TRUE(high group) = Dysfunction Score, FALSE(low group) = Exclusion Score]

############################################
##### Cross validation grid search #########

# Hyperparameter tuning and Estimate performance

# Target values [TIDE Score (Dysfunction Score or Exclusion Score)]

print(" ")
print("## 2. linear Regression ##")
print(" ")
#Top_list = ['top1', 'top2']
#TIDE_list = ['tide', 'dys', 'exc']
#total = ['ctl','tide','dys','exc']

for j in Top_list:
    for i in TIDE_list:
        # X and Y data load
        X_train = globals()["{}_x_train_{}".format(j,i)]
        X_test = globals()["{}_x_test_{}".format(j,i)]
        Y_train = globals()["y_train_{}".format(i)]
        Y_test = globals()["y_test_{}".format(i)]
        
        # Machine learning load
        LR = linear_model.LinearRegression(n_jobs=50)
        LR.fit(X_train, Y_train.values.ravel())

        # 10 fold cross validation
        LR_score = cross_val_score(LR, X_train, Y_train.values.ravel(), cv=10,
                                            scoring='neg_mean_squared_error')

        #print("Linear Regression NMSE[10 cv]:", LR_score.mean())

        # Model prediction
        globals()["{}_best_model_{}".format(j,i)] = LR
        globals()["{}_best_model_{}_pred".format(j,i)] = LR.predict(X_test)
        
        # Score
        globals()["{}_MSE_{}_test".format(j,i)] = mean_squared_error(Y_test, globals()["{}_best_model_{}_pred".format(j,i)])
        print("{}_MSE_{}_Score_test:".format(j,i) , globals()["{}_MSE_{}_test".format(j,i)]) 
        print(" ") 





#########################################################
########### Target Values Scatter plot ###############
########################################################

#Top_list = ['top1', 'top2']
#TIDE_list = ['tide', 'dys', 'exc']
#total = ['ctl','tide','dys','exc']

for n in Top_list:
    for i in TIDE_list:
        # data set
        obse = globals()["y_test_{}".format(i)]
        pred = pd.DataFrame(globals()["{}_best_model_{}_pred".format(n,i)])
        
        if i == 'dys':
            j = "Dysfunction"
            
        elif i == 'exc':
            j = "Exclusion"
        
        #elif i == 'tide':
        #    j = "TIDE"
        
        obse.columns=[j]
        pred.columns=[j]
        
        # Scatter plot
        plt.scatter(obse[j], pred[j], s=3)
    
        #regression line
        m, b = np.polyfit(obse[j], pred[j], 1)
        plt.plot(obse[j], m*obse[j]+b, color='red')

        # plot design
        plt.xlabel('Observed {}'.format(j) , fontsize = 14)
        plt.ylabel('Predicted {}'.format(j), fontsize = 14)

        # save plot
        plt.savefig('/home/dong/python_work/TIDE/Figure/SHAP_Top_miRNA/Scatter plot of {} linear {}.png'.format(n,j), dpi=600)

        # Clear reset plot
        plt.clf() 
        
        # Pearson Correlation [Regression line]
        pcc_pred = pd.DataFrame(globals()["{}_best_model_{}_pred".format(n,i)])
        pcc_pred.columns=[j]
        pcc_pred.reset_index(drop=True, inplace=True)
        
        pcc_obse = pd.DataFrame(globals()["y_test_{}".format(i)], columns=[j])
        pcc_obse.reset_index(drop=True, inplace=True)
        
        print("{} {} linear PCC:".format(n,j), pcc_obse[j].corr(pcc_pred[j]))
        
        


###############################################
##########  Prediction Result merge #########
###############################################

# CTL.flag [TRUE(high group) = Dysfunction Score, FALSE(low group) = Exclusion Score]

# Prediction Results split

#Top_list = ['top6', 'top4']
#TIDE_list = ['tide', 'dys', 'exc']
#total = ['ctl','dys','exc']

for n in Top_list:
    for m in total:
        if m == 'ctl':
            k = 'CTL.flag'
            globals()["{}_{}_pred_test".format(n,m)] = pd.DataFrame(globals()["{}_best_model_{}_pred".format(n,m)], columns = [k])
            
        elif m == 'dys':
            k = 'Dysfunction_pred'
            q = 'Dysfunction'
            
            globals()["{}_{}_pred_test".format(n,m)] = pd.DataFrame(globals()["{}_best_model_{}_pred".format(n,m)], columns = [k])
            
            globals()["{}_{}_test".format(n,m)] = pd.DataFrame(globals()["y_test_{}".format(m)], columns = [q])
            globals()["{}_{}_test".format(n,m)].reset_index(drop=True, inplace=True)
        
        elif m == 'exc':
            k = 'Exclusion_pred'
            q = 'Exclusion'
            
            globals()["{}_{}_pred_test".format(n,m)] = pd.DataFrame(globals()["{}_best_model_{}_pred".format(n,m)], columns = [k])
            
            globals()["{}_{}_test".format(n,m)] = pd.DataFrame(globals()["y_test_{}".format(m)], columns = [q])
            globals()["{}_{}_test".format(n,m)].reset_index(drop=True, inplace=True)
        

for n in Top_list:
    globals()["{}_total_pred_test".format(n)] = pd.concat([globals()["{}_ctl_pred_test".format(n)],
                                                           globals()["{}_dys_pred_test".format(n)],
                                                           globals()["{}_dys_test".format(n)],
                                                           globals()["{}_exc_pred_test".format(n)],
                                                           globals()["{}_exc_test".format(n)]], axis=1)
        
# merge

for n in Top_list:
    total_pred_test = globals()["{}_total_pred_test".format(n)]
    
    # CTL High -> Dysfunction Score
    globals()["{}_total_pred_test_dys".format(n)] = total_pred_test[total_pred_test['CTL.flag'] == True]
    globals()["{}_pred_dys".format(n)] = globals()["{}_total_pred_test_dys".format(n)][['Dysfunction_pred']]
    globals()["{}_test_dys".format(n)] = globals()["{}_total_pred_test_dys".format(n)][['Dysfunction']]
    
    # CTL Low -> Exclusion Score
    globals()["{}_total_pred_test_exc".format(n)] = total_pred_test[total_pred_test['CTL.flag'] == False]
    globals()["{}_pred_exc".format(n)] = globals()["{}_total_pred_test_exc".format(n)][['Exclusion_pred']]
    globals()["{}_test_exc".format(n)] = globals()["{}_total_pred_test_exc".format(n)][['Exclusion']]    
                                                           
                                                               
##########################
# Final TIDE Score

# Prediction Values
for n in Top_list:
    globals()["{}_pred_tide_dys".format(n)] = globals()["{}_pred_dys".format(n)]
    globals()["{}_pred_tide_dys".format(n)].rename(columns={"Dysfunction_pred": "TIDE_pred"}, inplace=True)
 
    globals()["{}_pred_tide_exc".format(n)] = globals()["{}_pred_exc".format(n)]
    globals()["{}_pred_tide_exc".format(n)].rename(columns={"Exclusion_pred": "TIDE_pred"}, inplace=True)
    
    globals()["{}_pred_tide".format(n)] = pd.concat([globals()["{}_pred_tide_dys".format(n)], 
                                                     globals()["{}_pred_tide_exc".format(n)]], axis=0)
    
# Observed Values
for n in Top_list:
    globals()["{}_test_tide_dys".format(n)] = globals()["{}_test_dys".format(n)]
    globals()["{}_test_tide_dys".format(n)].rename(columns={"Dysfunction": "TIDE"}, inplace=True)
    
    globals()["{}_test_tide_exc".format(n)] = globals()["{}_test_exc".format(n)]
    globals()["{}_test_tide_exc".format(n)].rename(columns={"Exclusion": "TIDE"}, inplace=True)
    
    globals()["{}_test_tide".format(n)] = pd.concat([globals()["{}_test_tide_dys".format(n)], 
                                                     globals()["{}_test_tide_exc".format(n)]], axis=0) 
                                                     


#############################################
#  Evaluate Prediction Results
print(" ")
print("## 3. Multi Model Evaluate Prediction Results ##")
print(" ")

for n in Top_list:
    for m in TIDE_list:
        globals()["{}_total_MSE_{}_test".format(n,m)] = mean_squared_error(globals()["{}_pred_{}".format(n,m)], globals()["{}_test_{}".format(n,m)])
        print("{}_Multi_MSE_{}_test:".format(n,m) , globals()["{}_total_MSE_{}_test".format(n,m)]) 
        print(" ")




########################################################
########### Multi Model Target Values Scatter plot #####
########################################################

for n in Top_list:
    for i in TIDE_list:
        # data set
        obse = globals()["{}_test_{}".format(n,i)]
        pred = globals()["{}_pred_{}".format(n,i)]
            
        #if i == 'dys':
        #    j = "Dysfunction"
            
        #elif i == 'exc':
        #    j = "Exclusion"
                
        if i == 'tide':
            j = "TIDE"
    
        obse.columns=[j]
        pred.columns=[j]
        
        # Scatter plot
        plt.scatter(obse[j], pred[j], s=3)
        
        #regression line
        m, b = np.polyfit(obse[j], pred[j], 1)
        plt.plot(obse[j], m*obse[j]+b, color='red')
        
        # plot design
        plt.xlabel('Observed {}'.format(j) , fontsize = 14)
        plt.ylabel('Predicted {}'.format(j), fontsize = 14)
        
        # save plot
        plt.savefig('/home/dong/python_work/TIDE/Figure/SHAP_Top_miRNA/Scatter plot of {} Multi linear {}.png'.format(n,j), dpi=1200)

        # Clear reset plot
        plt.clf() 
        
        # Pearson Correlation [Regression line]
        pcc_pred = pd.DataFrame(globals()["{}_pred_{}".format(n,i)])
        pcc_pred.columns=[j]
        pcc_pred.reset_index(drop=True, inplace=True)
        
        pcc_obse = pd.DataFrame(globals()["{}_test_{}".format(n,i)], columns=[j])
        pcc_obse.reset_index(drop=True, inplace=True)
        
        print("{} Multi {} linear PCC:".format(n,j), pcc_obse[j].corr(pcc_pred[j]))






#########################################################
############### Validation Test #########################
#########################################################

print(" ")
print("#########################################################")
print("############### Validation Test #########################")
print("#########################################################")
print(" ")


# Data load
val_data = pd.read_csv("/home/dong/python_work/TIDE/final_TIDE_Score/validation/TIDE_Score_TCGA_Val_miR_rm_dup.csv")

x_val = val_data.iloc[:,2:1883]

### Standard Scaling ###

scaler_val = StandardScaler()

x_val_list = list(x_val.columns)

x_val_sc = x_val.copy()
x_val_sc[x_list] = scaler.fit_transform(x_val[x_val_list])



# Target Values


# Top 1 ( mean(SHAP value) > 0.01 )

top1_x_val_ctl = x_val_sc[['hsa.mir.155','hsa.mir.150','hsa.mir.4772']]

top1_x_val_tide = x_val_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183','hsa.mir.155','hsa.mir.345']]

top1_x_val_dys = x_val_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a','hsa.mir.210']]

top1_x_val_exc = x_val_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150','hsa.mir.192',
                               'hsa.mir.141','hsa.mir.155','hsa.mir.200c','hsa.mir.493','hsa.mir.708','hsa.mir.142']]


# Top 2 ( mean(SHAP value) > 0.02 )

top2_x_val_ctl = x_val_sc[['hsa.mir.150','hsa.mir.155']]

top2_x_val_tide = x_val_sc[['hsa.mir.10b','hsa.mir.150','hsa.mir.10a','hsa.mir.183']]

top2_x_val_dys = x_val_sc[['hsa.mir.10b','hsa.mir.183','hsa.mir.150','hsa.mir.151a']]

top2_x_val_exc = x_val_sc[['hsa.mir.10b','hsa.mir.194.1','hsa.mir.10a','hsa.mir.194.2','hsa.mir.150']]




y_val_ctl = val_data[['CTL.flag']]
#print(y_val_ctl)

y_val_tide = val_data[['TIDE']]
#print(y_val_exc)

y_val_dys = val_data[['Dysfunction']]
#print(y_val_dys)

y_val_exc = val_data[['Exclusion']]
#print(y_val_exc)




# Validation Test Classification Model [CTL group]

print("# Validation Test Classification Model (Logistic) [CTL group]")
print(" ")

for n in Top_list:
    # Prediction
    best_model_ctl = globals()["{}_best_model_ctl".format(n)]
    globals()["{}_LRC_ctl_val_pred".format(n)] = best_model_ctl.predict(globals()["{}_x_val_ctl".format(n)])
    
    # Score
    globals()["{}_F1_CTL_val".format(n)] = f1_score(y_val_ctl, globals()["{}_LRC_ctl_val_pred".format(n)])
    globals()["{}_Balanced_AUC_CTL_val".format(n)] = balanced_accuracy_score(y_val_ctl, globals()["{}_LRC_ctl_val_pred".format(n)])

    print("{}_F1_Score_CTL:".format(n), globals()["{}_F1_CTL_val".format(n)])
    print("{}_Balanced_AUC_CTL:".format(n), globals()["{}_Balanced_AUC_CTL_val".format(n)])

    # ROCAUC Plot
    globals()["{}_visualizer_val".format(n)] = ROCAUC(best_model_ctl)
    globals()["{}_visualizer_val".format(n)].fit(globals()["{}_x_train_ctl".format(n)], y_train_ctl.values.ravel())

    globals()["{}_visualizer_val".format(n)].score(globals()["{}_x_val_ctl".format(n)], y_val_ctl.values.ravel())
    globals()["{}_visualizer_val".format(n)].show(outpath="/home/dong/python_work/TIDE/Figure/SHAP_Top_miRNA/ROC_Curves_{}_val_CTL_Logi.png".format(n),dpi=1200)

    # Clear reset plot
    plt.clf() 


# Validation Test Regression Model [TIDE, Dysfunction, Exclusion]

for n in Top_list:
    for i in TIDE_list:
        print(" ")
        print("# {} Validation Test Regression Model (Linear) [{}]".format(n,i))
    
        # predict
        best_model_val = globals()["{}_best_model_{}".format(n,i)]
        val_pred = best_model_val.predict(globals()["{}_x_val_{}".format(n,i)])
        globals()["{}_linear_{}_val_pred".format(n,i)] = val_pred
    
        # Estimate
        mse_val = mean_squared_error(globals()["y_val_{}".format(i)], globals()["{}_linear_{}_val_pred".format(n,i)])
        globals()["{}_mse_linear_{}_val".format(n,i)] = mse_val
    
        print("{}_MSE_linear_{}_Val:".format(n,i), globals()["{}_mse_linear_{}_val".format(n,i)])



#########################################################
########### Target Values Scatter plot ###############
########################################################


for n in Top_list:
    for i in TIDE_list:
        if i == 'dys':
            j = "Dysfunction"
    
        elif i == 'exc':
            j = "Exclusion"
            
        #elif i == 'tide':
        #    j = "TIDE"
        
        # data set
        obse = globals()["y_val_{}".format(i)]
        pred = pd.DataFrame(globals()["{}_linear_{}_val_pred".format(n,i)])
    
        obse.columns=[j]
        pred.columns=[j]
    
        # Scatter plot
        plt.scatter(obse[j], pred[j], s=3)
    
        #regression line
        m, b = np.polyfit(obse[j], pred[j], 1)
        plt.plot(obse[j], m*obse[j]+b, color='red')

        # plot design
        plt.xlabel('Observed {}'.format(j) , fontsize = 14)
        plt.ylabel('Predicted {}'.format(j), fontsize = 14)
        
        # save plot
        plt.savefig('/home/dong/python_work/TIDE/Figure/SHAP_Top_miRNA/Scatter plot of {} linear Val {}.png'.format(n,j), dpi=600)

        # Clear reset plot
        plt.clf() 
        
        # Pearson Correlation [Regression line]
        pcc_pred = pd.DataFrame(globals()["{}_linear_{}_val_pred".format(n,i)])
        pcc_pred.columns=[j]
        pcc_pred.reset_index(drop=True, inplace=True)
        
        pcc_obse = pd.DataFrame(globals()["y_val_{}".format(i)], columns=[j])
        pcc_obse.reset_index(drop=True, inplace=True)
        
        print(" ")
        print("{} {} Linear PCC [val]:".format(n,j), pcc_obse[j].corr(pcc_pred[j]))




########################################################
###### Validation Multi Model Results processing #######

# Prediction Results split
for n in Top_list:
    for m in total:
        # CTL
        if m == 'ctl':
            k = 'CTL.flag'
            
            globals()["{}_{}_pred_val".format(n,m)] = pd.DataFrame(globals()["{}_LRC_{}_val_pred".format(n,m)], columns = [k])
        
        # Dysfunction
        elif m == 'dys':
            k = 'Dysfunction_pred'
            q = 'Dysfunction'
            
            globals()["{}_{}_pred_val".format(n,m)] = pd.DataFrame(globals()["{}_linear_{}_val_pred".format(n,m)], columns = [k])
            
            globals()["{}_{}_val".format(n,m)] = pd.DataFrame(globals()["y_val_{}".format(m)], columns = [q])
            globals()["{}_{}_val".format(n,m)].reset_index(drop=True, inplace=True)
              
        # Exclusion    
        elif m == 'exc':
            k = 'Exclusion_pred'
            q = 'Exclusion'
            
            globals()["{}_{}_pred_val".format(n,m)] = pd.DataFrame(globals()["{}_linear_{}_val_pred".format(n,m)], columns = [k])
            
            globals()["{}_{}_val".format(n,m)] = pd.DataFrame(globals()["y_val_{}".format(m)], columns = [q])
            globals()["{}_{}_val".format(n,m)].reset_index(drop=True, inplace=True)


# combine
for n in Top_list:
    globals()["{}_total_pred_val".format(n)] = pd.concat([globals()["{}_ctl_pred_val".format(n)],
                                                           globals()["{}_dys_pred_val".format(n)],
                                                           globals()["{}_dys_val".format(n)],
                                                           globals()["{}_exc_pred_val".format(n)],
                                                           globals()["{}_exc_val".format(n)]], axis=1)
        

for n in Top_list:
    total_pred_val = globals()["{}_total_pred_val".format(n)]
    
    # CTL High -> Dysfunction Score
    globals()["{}_total_pred_val_dys".format(n)] = total_pred_val[total_pred_val['CTL.flag'] == True]
    
    globals()["{}_pred_dys_val".format(n)] = globals()["{}_total_pred_val_dys".format(n)][['Dysfunction_pred']]
    globals()["{}_val_dys".format(n)] = globals()["{}_total_pred_val_dys".format(n)][['Dysfunction']]
    
    # CTL Low -> Exclusion Score
    ctl_val = pd.DataFrame(y_val_ctl, columns = ['CTL.flag'])
    globals()["{}_total_pred_val_exc".format(n)] = pd.concat([ctl_val, 
                                                              globals()["{}_exc_pred_val".format(n)], 
                                                              globals()["{}_exc_val".format(n)]], axis=1)
    
    globals()["{}_total_pred_val_exc".format(n)] = globals()["{}_total_pred_val_exc".format(n)][globals()["{}_total_pred_val_exc".format(n)]['CTL.flag'] == False]
    
    globals()["{}_pred_exc_val".format(n)] = globals()["{}_total_pred_val_exc".format(n)][['Exclusion_pred']]
    globals()["{}_val_exc".format(n)] = globals()["{}_total_pred_val_exc".format(n)][['Exclusion']]    
                                                           
                                                          


# TIDE Score 

# Prediction Values
for n in Top_list:
    globals()["{}_pred_tide_val_dys".format(n)] = globals()["{}_pred_dys_val".format(n)]
    globals()["{}_pred_tide_val_dys".format(n)].rename(columns={"Dysfunction_pred": "TIDE_pred"}, inplace=True)
 
    globals()["{}_pred_tide_val_exc".format(n)] = globals()["{}_pred_exc_val".format(n)]
    globals()["{}_pred_tide_val_exc".format(n)].rename(columns={"Exclusion_pred": "TIDE_pred"}, inplace=True)
    
    globals()["{}_pred_tide_val".format(n)] = pd.concat([globals()["{}_pred_tide_val_dys".format(n)], 
                                                     globals()["{}_pred_tide_val_exc".format(n)]], axis=0)
    
# Observed Values
for n in Top_list:
    globals()["{}_test_tide_val_dys".format(n)] = globals()["{}_val_dys".format(n)]
    globals()["{}_test_tide_val_dys".format(n)].rename(columns={"Dysfunction": "TIDE"}, inplace=True)
    
    globals()["{}_test_tide_val_exc".format(n)] = globals()["{}_val_exc".format(n)]
    globals()["{}_test_tide_val_exc".format(n)].rename(columns={"Exclusion": "TIDE"}, inplace=True)
    
    globals()["{}_val_tide".format(n)] = pd.concat([globals()["{}_test_tide_val_dys".format(n)], 
                                                     globals()["{}_test_tide_val_exc".format(n)]], axis=0) 




#############################################
# Validation Multi Model Prediction Results
#  Evaluate Prediction Results

print("## Validation Multi Model Evaluate Prediction Results ##")

for n in Top_list:
    for m in TIDE_list:
        globals()["{}_Multi_MSE_{}_val".format(n,m)] = mean_squared_error(globals()["{}_val_{}".format(n,m)], globals()["{}_pred_{}_val".format(n,m)])
        print("{}_Multi_Model_MSE_{}_val:".format(n,m) , globals()["{}_Multi_MSE_{}_val".format(n,m)]) 

print(" ")

######################################################
########### Validation data Scatter plot #############
######################################################

for n in Top_list:
    for i in TIDE_list:
        
        #if i == 'dys':
        #    j = "Dysfunction"
            
        #elif i == 'exc':
        #    j = "Exclusion"
                
        if i == 'tide':
            j = "TIDE"
    
        # data set
        obse = globals()["{}_val_{}".format(n,i)]
        pred = globals()["{}_pred_{}_val".format(n,i)]
    
        obse.columns=[j]
        pred.columns=[j]
    
        # Scatter plot
        plt.scatter(obse[j], pred[j], s=3)
        
        #regression line
        m, b = np.polyfit(obse[j], pred[j], 1)
        plt.plot(obse[j], m*obse[j]+b, color='red')
        
        # plot design
        plt.xlabel('Observed {}'.format(j) , fontsize = 14)
        plt.ylabel('Predicted {}'.format(j), fontsize = 14)
        
        # save plot
        plt.savefig('/home/dong/python_work/TIDE/Figure/SHAP_Top_miRNA/Scatter plot of {} Multi Val {}.png'.format(n,j),dpi=1200)

        # Clear reset plot
        plt.clf() 
        
        # Pearson Correlation [Regression line]
        pcc_pred = pd.DataFrame(globals()["{}_pred_{}_val".format(n,i)])
        pcc_pred.columns=[j]
        pcc_pred.reset_index(drop=True, inplace=True)
        
        pcc_obse = pd.DataFrame(globals()["{}_val_{}".format(n,i)], columns=[j])
        pcc_obse.reset_index(drop=True, inplace=True)
        
        print("{} {} Multi PCC [val]:".format(n,j), pcc_obse[j].corr(pcc_pred[j]))

print(" ")




#Import data as pandas conversion into data frame Step 1
import pandas as pd 
data = pd.read_csv('C:/Users/Harve/Videos/Project/Data.csv')

#Step 2: Data visulization


#Statistical Analysis 

print(data.head())

print(data.columns)

print(data['X'])

print(data['Y'])

print(data['Z'])

#Visualize data
import numpy as np
import matplotlib.pyplot as mp


#Decribe datset
print(data.describe())

#Histograms for Data X, Y and Z

data['X'].hist()
data['Y'].hist()
data['Z'].hist()

#Each data set
data.hist(bins=30)
 

#Stratfied Splitter



X = data[["X", "Y", "Z"]]
y = data[["Step"]]

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Collinear Matrix

corr_matrix = data[["X", "Y", "Z", "Step"]].corr(method = "pearson")
print(corr_matrix)

#plot the correlation matrix

import seaborn as sns
sns.heatmap(np.abs(corr_matrix))

mp.figure(figsize=(10,8))   # size of the figure
sns.heatmap(corr_matrix, 
            annot=True,      # show numbers inside cells
            cmap="coolwarm", # color scheme
            fmt=".2f",       # format to 2 decimals
            square=True)     # force square cells
mp.title("Correlation Matrix (Pearson)")
mp.show()

print("\n")


#Step 4 classfification Model Development/Engineering

#grid Search 

#Model Selection: Logistic Regression

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression


param_grid_logistic_regression = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "solver": ['liblinear', 'saga']
    }

log_regression = LogisticRegression(max_iter=10000)
grid_Logistic = GridSearchCV(estimator = log_regression, 
                             param_grid= param_grid_logistic_regression, 
                             scoring = 'accuracy',
                             n_jobs= -1,
                             cv = 5)

grid_Logistic.fit(X_train, y_train.values.ravel())

print("Best Parameters for Logstic Regression:", grid_Logistic.best_params_)
print("Best Scores for Logstics Regression:", grid_Logistic.best_score_)
print("\n")



#Model Selection: Random Forests


from sklearn.ensemble import RandomForestClassifier

paramgird_RandomForest = {
    
    "n_estimators": [100, 200, 500],
    "max_depth" : [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
    
    }

randomForestCL = RandomForestClassifier(random_state=42)

GridSearch_RF = GridSearchCV(estimator=randomForestCL,
                             param_grid= paramgird_RandomForest, 
                             scoring = 'accuracy', 
                             n_jobs=-1,
                             cv = 5)
GridSearch_RF.fit(X_train, y_train.values.ravel())

print("Best Parameter for RandomForestClassfifer", GridSearch_RF.best_params_)
print("Best scores for Random Forest classifier", GridSearch_RF.best_score_)
print("\n")

#SVM

from sklearn.svm import SVC

param_SVM_Grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma":[0.001, 0.01, 0.1, 1]
    }
gridSearch_SVM = GridSearchCV(SVC(),
                              param_grid=param_SVM_Grid, 
                              cv = 5,
                              scoring = 'accuracy')

gridSearch_SVM.fit(X_train, y_train.values.ravel())

print("BEst Parameter for SVM:", gridSearch_SVM.best_params_)
print("Best Scores for SVM", gridSearch_SVM.best_score_)
print("\n")


#RandomizedSearchCV with SVM

from sklearn.model_selection import RandomizedSearchCV

RandomizedSearch_RF= RandomizedSearchCV(estimator = randomForestCL, 
                                        param_distributions = paramgird_RandomForest,
                                        scoring = 'accuracy',
                                        n_jobs = -1,
                                        cv = 5)
RandomizedSearch_RF.fit(X_train, y_train.values.ravel())

print("Best parameter for RandomForest Randomizer Serach", RandomizedSearch_RF.best_params_ )
print("Best score for RandomForest Randomizer Serach", RandomizedSearch_RF.best_score_)
print("\n")


from tabulate import tabulate

data_table = [["logistic Regression", grid_Logistic.best_params_, grid_Logistic.best_score_],
              ["RainForest", GridSearch_RF.best_params_, GridSearch_RF.best_score_],
              ["SVM", gridSearch_SVM.best_params_, gridSearch_SVM.best_score_],
              ["Randomized Search", RandomizedSearch_RF.best_params_, RandomizedSearch_RF.best_score_]]

headers = ["Model", "Best Parameters", "Best Score"]

print(tabulate(data_table, headers=headers, tablefmt = "grid"))
print("\n")

#Step 5: Stacked Model Performance 

#Redfine

BestLogisitics_Model = grid_Logistic.best_estimator_
BestRandomForest_Model = GridSearch_RF.best_estimator_
BestSVM_Model = gridSearch_SVM.best_estimator_
BestRandomizedRF_Model = RandomizedSearch_RF.best_estimator_


#Put every model in array 

from sklearn.metrics import f1_score, accuracy_score, precision_score

Models = {
    "Logistic": BestLogisitics_Model,
    "RandomForest": BestRandomForest_Model,
    "SVM": BestSVM_Model,
    "RandomizedRF": BestRandomizedRF_Model
}

results = {}

for name, model in Models.items():
    y_pred = model.predict(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    }

y_pred_log = model.predict(X_test)

print(results)
print("\n")


results_df = pd.DataFrame(results).T  # Transpose to make models the rows

# Optionally round for cleaner display
results_df = results_df.round(4)

print(results_df)


#Create Confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#True and predicted values (y_test and y_pred)

cm = confusion_matrix(y_test, y_pred_log)

#Display the confusion matrix 

labels = sorted(list(set(y_pred_log)))

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels)
disp.plot(cmap = mp.cm.Blues)
mp.title('Confusion Matrix Logistics Regression')
mp.xlabel('Predicted Label Logistics')
mp.ylabel('Test Label')
mp.show()


#StackingClassifier 

estimators = [('lr', BestLogisitics_Model),
              ('svc', BestSVM_Model)
             ]


from sklearn.ensemble import StackingClassifier
stackingClass = StackingClassifier(estimators=estimators,
                              final_estimator=BestRandomForest_Model)
        
stackingClass.fit(X_train,y_train)
predictions = stackingClass.predict(X_test)

#Extract Accuracy, precesion, and F1

Accuracy_prediction = accuracy_score(y_true = y_test, y_pred = predictions)
Precesion_prediction = precision_score(y_true = y_test,y_pred = predictions, average='weighted')
F1_score_prediction = f1_score(y_true=y_test, y_pred=predictions, average='weighted')


print("\n")
print("Accuracy of Stacker", Accuracy_prediction)
print("\n")
print("Precesion of Stacker", Precesion_prediction)
print("\n")
print("F1 score of Stacker", F1_score_prediction)
print("\n")

cm_stacker = confusion_matrix(y_test, y_pred=predictions)

labels_Stacker = sorted(list(set(predictions)))

disp = ConfusionMatrixDisplay(confusion_matrix = cm_stacker, display_labels=labels_Stacker)
disp.plot(cmap = mp.cm.Blues)
mp.title('Confusion Matrix Stacker')
mp.xlabel('Predicted Label Stacker')
mp.ylabel('Test Label')
mp.show()

#Step 6

#Joblib 

from joblib import dump, load 

model_log = BestLogisitics_Model
dump(model_log, 'logistics_regression_model.joblib')

loaded_model = load('logistics_regression_model.joblib')

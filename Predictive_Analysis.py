import pandas as pd
import numpy as np
import math

from sklearn import metrics, model_selection, linear_model, neighbors, ensemble, preprocessing

#%% Downloading all data from CSV files to DataFrames

city_adm=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/city_adm.csv",sep=",")
city_loc=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/city_loc.csv",sep=",")
city_pop=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/city_pop.csv",sep=",")
code_DEGREE=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_DEGREE.csv",sep=",")
code_economic_sector=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_economic_sector.csv",sep=",")
code_employee_count=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_employee_count.csv",sep=",")
code_Employer_type=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_Employer_type.csv",sep=",")
code_Household=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_Household.csv",sep=",")
code_JOB_CATEGORY=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_JOB_CATEGORY.csv",sep=",")
code_Job_condition=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_Job_condition.csv",sep=",")
code_job_desc=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_job_desc.csv",sep=",")
code_job_desc_map=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_job_desc_map.csv",sep=",")
code_job_desc_n1=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_job_desc_n1.csv",sep=",")
code_job_desc_n2=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_job_desc_n2.csv",sep=",")
code_TYPE_OF_CONTRACT=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/code_TYPE_OF_CONTRACT.csv",sep=",")
departments=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/departments.csv",sep=",")
regions=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/regions.csv",sep=",")
learn=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/learn.csv",sep=",")
test=pd.read_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/project-15-files/test.csv",sep=",")

#%% Looking for and counting missing values in learn and test

print(learn.isnull().sum())
print(test.isnull().sum())

print(learn[learn["Working_hours"].isnull()]["Working_hours"])
print(test[test["Working_hours"].isnull()]["Working_hours"])

#%% Filling missing values in "Working hours" and checking the result

learn["Working_hours"]=learn["Working_hours"].fillna(learn["Working_hours"].median())
test["Working_hours"]=test["Working_hours"].fillna(learn["Working_hours"].median())

print(learn.info())
print(test.info())

#%% Merging informative datasets on job descriptions, cities, departments and regions

CSP=pd.merge(code_job_desc_map,code_job_desc,left_on="N3",right_on="Code").drop("Code",axis=1)
CSP=pd.merge(CSP,code_job_desc_n2,left_on="N2",right_on="Code",suffixes=("_N3","")).drop("Code",axis=1)
CSP=pd.merge(CSP,code_job_desc_n1,left_on="N1",right_on="Code",suffixes=("_N2","_N1")).drop("Code",axis=1)

city=pd.merge(city_adm,city_loc)
city=pd.merge(city,city_pop)

geo=pd.merge(departments,regions)

#%% Completing learn dataset with information provided by the three previous ones

learn_complete=pd.merge(learn,CSP,left_on="job_desc",right_on="N3").drop(["job_desc","N3","N1","Libellé_N1","Libellé_N2","Libellé_N3"],axis=1)
learn_complete=pd.merge(learn_complete,geo,left_on="job_dep",right_on="DEP").drop("DEP",axis=1)
learn_complete=pd.merge(learn_complete,city)
print(learn_complete)

# Solving the Corsican problem by merging the two departments into one (N°20)
learn_complete[(learn_complete["DEP"]=="2A") | (learn_complete["DEP"]=="2B")]=20
learn_complete[(learn_complete["job_dep"]=="2A") | (learn_complete["job_dep"]=="2B")]=20

# Converting "object" types in "category" types
L1=learn_complete.select_dtypes(include='object').columns
for i in L1:
    learn_complete[i]=learn_complete[i].astype("category")
    
print(learn_complete.dtypes)
    
#%% Completing test dataset with information provided by the three previous ones

test_complete=pd.merge(test,CSP,left_on="job_desc",right_on="N3").drop(["job_desc","N3","N1","Libellé_N1","Libellé_N2","Libellé_N3"],axis=1)
test_complete=pd.merge(test_complete,geo,left_on="job_dep",right_on="DEP").drop("DEP",axis=1)
test_complete=pd.merge(test_complete,city)
print(test_complete)

# Solving the Corsican problem by merging the two departments into one (N°20)
test_complete[(test_complete["DEP"]=="2A") | (test_complete["DEP"]=="2B")]=20
test_complete[(test_complete["job_dep"]=="2A") | (test_complete["job_dep"]=="2B")]=20

# Converting "object" types in "category" types
L2=test_complete.select_dtypes(include='object').columns
for i in L2:
    test_complete[i]=test_complete[i].astype("category")
    
print(test_complete.dtypes)
    
#%% One-hot encoding learning and test sets

# N2 will be used for job descriptions 
learn_data=pd.get_dummies(learn_complete, columns=["Sex","Household","DEGREE","TYPE_OF_CONTRACT","economic_sector","employee_count","Job_condition","Employer_type","JOB_CATEGORY","N2","Town_type"])
learn_data=learn_data.drop(["INSEE","Emolument","Nom de la commune","Nom du département","Nom de la région"],axis=1)
learn_label=learn_complete["Emolument"]

test_data=pd.get_dummies(test_complete, columns=["Sex","Household","DEGREE","TYPE_OF_CONTRACT", "economic_sector","employee_count","Job_condition","Employer_type","JOB_CATEGORY","N2","Town_type"])
test_data=test_data.drop(["INSEE","Nom de la commune","Nom du département","Nom de la région"],axis=1)

#%%

learn_data=preprocessing.scale(learn_data)
test_data=preprocessing.scale(test_data)

#%% Machine learning algorithms

#%% Predicting with linear regression

lin = linear_model.LinearRegression()
lin.fit(learn_data,learn_label)
print("R² : ",lin.score(learn_data,learn_label))

scores=model_selection.cross_val_score(lin,learn_data,learn_label)
print("Cross-validation score : ",scores.mean(),scores.std())

lin_predict=pd.Series(lin.predict(test_data))

#%% Predicting with Ridge

ridge = linear_model.Ridge()
ridge.fit(learn_data, learn_label)
print("R² : ",ridge.score(learn_data,learn_label))

scores=model_selection.cross_val_score(ridge,learn_data,learn_label,cv=5)
print("Cross-validation score : ",scores.mean(),scores.std())

ridge_predict=pd.Series(ridge.predict(test_data))

#%% Predicting with KNeighborsRegressor (quite long ~ 10min)

knn_cv=[]
knn_train=[]
ks=np.array(range(1,25,2))
for k in ks:
    knn_algo=neighbors.KNeighborsRegressor(k)
    knn_cv.append(-model_selection.cross_val_score(knn_algo,learn_data,learn_label,cv=5,scoring="neg_mean_squared_error"))
    knn_algo.fit(learn_data,learn_label)
    knn_train.append(metrics.mean_squared_error(learn_label,knn_algo.predict(learn_data)))
    
knn_mse=np.array([x.mean() for x in knn_cv])
knn_mse_std=np.array([x.std() for x in knn_cv])/math.sqrt(5)
best_acc_pos=knn_mse.argmin()
best_k=ks[best_acc_pos]

print("Best k :",best_k,", with mean squared error :",knn_mse[best_acc_pos])

knr = neighbors.KNeighborsRegressor(best_k)
knr.fit(learn_data, learn_label)
print("R² : ",knr.score(learn_data,learn_label))

scores=model_selection.cross_val_score(knr,learn_data,learn_label,cv=5)
print("Cross-validation score : ",scores.mean(),scores.std())

knr_predict=pd.Series(knr.predict(test_data))

#%% Predicting with random forest regressor

rf = ensemble.RandomForestRegressor(n_estimators=60, max_depth=25, min_samples_split=20, n_jobs=2, max_features=30, oob_score=True)
rf.fit(learn_data,learn_label)
print("R² : ",rf.score(learn_data,learn_label))

scores=model_selection.cross_val_score(rf,learn_data,learn_label)
print("Cross-validation score : ",scores.mean(),scores.std())
print("Out-of-bag score : ",rf.oob_score_)

rf_predict=pd.Series(rf.predict(test_data))

#%% Predicting with gradient boosting regressor

gd = ensemble.GradientBoostingRegressor(n_estimators=60, max_depth=5, loss='ls', verbose=5)
gd.fit(learn_data,learn_label)
print("R² : ",gd.score(learn_data,learn_label))

scores=model_selection.cross_val_score(gd,learn_data,learn_label)
print("Cross-validation score : ",scores.mean(),scores.std())

gd_predict=pd.Series(gd.predict(test_data))

#%% Loading the prediction with the highest cross-validation score to a CSV file

gd_predict.to_csv("C:/Users/Loupicha/Documents/M2/Machine_Learning/project-15/prediction.csv",header=["Emolument"],index=False)

#%% Quick analysis of the 2 best predictions

print(rf_predict.describe())
print(gd_predict.describe())









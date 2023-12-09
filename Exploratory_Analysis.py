import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from kneed import KneeLocator # needs installation of kneed module
import scipy.cluster.hierarchy as sch

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
test["Working_hours"]=test["Working_hours"].fillna(test["Working_hours"].median())

print(learn.info())
print(test.info())

#%% Simple descriptive analysis

print(learn.shape)
print(learn.describe())

print(test.shape)
print(test.describe())

#%% Merging informative datasets on job descriptions, cities, departments and regions

CSP=pd.merge(code_job_desc_map,code_job_desc,left_on="N3",right_on="Code").drop("Code",axis=1)
CSP=pd.merge(CSP,code_job_desc_n2,left_on="N2",right_on="Code",suffixes=("_N3","")).drop("Code",axis=1)
CSP=pd.merge(CSP,code_job_desc_n1,left_on="N1",right_on="Code",suffixes=("_N2","_N1")).drop("Code",axis=1)

city=pd.merge(city_adm,city_loc)
city=pd.merge(city,city_pop)

geo=pd.merge(departments,regions)

#%% Completing learn and test datasets with information provided by the other ones

learn_complete=pd.merge(learn,CSP,left_on="job_desc",right_on="N3").drop("job_desc",axis=1)
learn_complete=pd.merge(learn_complete,geo,left_on="job_dep",right_on="DEP").drop("DEP",axis=1)
learn_complete=pd.merge(learn_complete,city)
learn_complete=pd.merge(learn_complete,code_DEGREE,left_on="DEGREE",right_on="Code").drop("Code",axis=1)
learn_complete=pd.merge(learn_complete,code_economic_sector,left_on="economic_sector",right_on="Code",suffixes=("_DEGREE","")).drop("Code",axis=1)
learn_complete=pd.merge(learn_complete,code_employee_count,left_on="employee_count",right_on="Code",suffixes=("_economic_sector","")).drop("Code",axis=1)
learn_complete=pd.merge(learn_complete,code_Employer_type,left_on="Employer_type",right_on="Code",suffixes=("_employee_count","")).drop("Code",axis=1)
learn_complete=pd.merge(learn_complete,code_Household,left_on="Household",right_on="Code",suffixes=("_Employer_type","")).drop("Code",axis=1)
learn_complete=pd.merge(learn_complete,code_JOB_CATEGORY,left_on="JOB_CATEGORY",right_on="Code",suffixes=("_Household","")).drop("Code",axis=1)
learn_complete=pd.merge(learn_complete,code_Job_condition,left_on="Job_condition",right_on="Code",suffixes=("_JOB_CATEGORY","")).drop("Code",axis=1)
learn_complete=pd.merge(learn_complete,code_TYPE_OF_CONTRACT,left_on="TYPE_OF_CONTRACT",right_on="Code",suffixes=("_Job_condition","_TYPE_OF_CONTRACT")).drop("Code",axis=1)
print(learn_complete.head(10))

test_complete=pd.merge(test,CSP,left_on="job_desc",right_on="N3").drop("job_desc",axis=1)
test_complete=pd.merge(test_complete,geo,left_on="job_dep",right_on="DEP").drop("DEP",axis=1)
test_complete=pd.merge(test_complete,city)
test_complete=pd.merge(test_complete,code_DEGREE,left_on="DEGREE",right_on="Code").drop("Code",axis=1)
test_complete=pd.merge(test_complete,code_economic_sector,left_on="economic_sector",right_on="Code",suffixes=("_DEGREE","")).drop("Code",axis=1)
test_complete=pd.merge(test_complete,code_employee_count,left_on="employee_count",right_on="Code",suffixes=("_economic_sector","")).drop("Code",axis=1)
test_complete=pd.merge(test_complete,code_Employer_type,left_on="Employer_type",right_on="Code",suffixes=("_employee_count","")).drop("Code",axis=1)
test_complete=pd.merge(test_complete,code_Household,left_on="Household",right_on="Code",suffixes=("_Employer_type","")).drop("Code",axis=1)
test_complete=pd.merge(test_complete,code_JOB_CATEGORY,left_on="JOB_CATEGORY",right_on="Code",suffixes=("_Household","")).drop("Code",axis=1)
test_complete=pd.merge(test_complete,code_Job_condition,left_on="Job_condition",right_on="Code",suffixes=("_JOB_CATEGORY","")).drop("Code",axis=1)
test_complete=pd.merge(test_complete,code_TYPE_OF_CONTRACT,left_on="TYPE_OF_CONTRACT",right_on="Code",suffixes=("_Job_condition","_TYPE_OF_CONTRACT")).drop("Code",axis=1)
print(test_complete.head(10))

#%% Concatenating the enhanced learn and test to have a more prcise analysis

learn_test_complete=pd.concat([learn_complete,test_complete])
print(learn_test_complete.head(10))

#%% Converting "object" types in "category" types

L=learn_test_complete.select_dtypes(include='object').columns
for i in L:
    learn_test_complete[i]=learn_test_complete[i].astype("category")
    
print(learn_test_complete.dtypes)



#%% Building the first dataset

dataset1=learn_test_complete[["Sex","Household","DEGREE","age_2020","DEP"]]
B1=pd.get_dummies(dataset1, columns=["Sex","Household","DEGREE"], prefix="", prefix_sep="")
dataset1=B1.pivot_table(index="DEP", values=B1.columns, aggfunc='mean').reset_index()
col1=B1.columns.tolist()
dataset1=dataset1[col1] # Useful to order the columns by the categories they are derived from
dataset1.set_index("DEP",inplace=True)
dataset1["inhabitants"]=city.groupby("DEP").aggregate(inhabitants=("inhabitants","sum"))
dataset1.reset_index(inplace=True) # Useful not to have an index, for later uses
print(dataset1.round(2))

#%% Building the second dataset 

dataset2=learn_test_complete[["TYPE_OF_CONTRACT", "economic_sector","employee_count","Job_condition","Employer_type","JOB_CATEGORY","job_dep","Working_hours","Emolument"]]
B2=pd.get_dummies(dataset2, columns=["TYPE_OF_CONTRACT", "economic_sector","employee_count","Job_condition","Employer_type","JOB_CATEGORY"], prefix="", prefix_sep="")
dataset2=B2.pivot_table(index="job_dep", values=B2.columns, aggfunc='mean').reset_index()
col2=B2.columns.tolist()
dataset2=dataset2[col2] # Useful to order the columns by the categories they are derived from # Useful not to have an index, for later uses
print(dataset2.round(2))

#%%

print(dataset1.isnull().sum())

print(dataset2.isnull().sum())
print(dataset2[dataset2["Emolument"].isnull()]["Emolument"])

#%% Filling missing values in "Working hours" and checking the result

dataset2["Emolument"]=dataset2["Emolument"].fillna(dataset2["Emolument"].median())

print(dataset2.info())

#%% Visualizing data from dataset 1

#%% Starting with standard deviation

print(dataset1.std())

#%% Age distribution

sns.histplot(data=dataset1,x="age_2020")

#%% Population distribution

sns.histplot(data=dataset1,x="inhabitants")

#%% Gender distribution

labels = ['Female', 'Male']
size = learn_test_complete['Sex'].value_counts()
colors = ['lightgreen', 'orange']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()

#%% Degree distribution

plt.rcParams['figure.figsize'] = (15, 8)
sns.countplot(x=learn_test_complete['DEGREE'], palette = 'hsv')
plt.title('Distribution of degree', fontsize = 20)
plt.show()

#%% Household distribution

plt.rcParams['figure.figsize'] = (15, 8)
sns.countplot(x=learn_test_complete['Household'], palette = 'hsv')
plt.title('Distribution of household', fontsize = 20)
plt.show()


#%% Mean age for each department

sns.lineplot(x = dataset1['DEP'], y= dataset1['age_2020'], color = 'blue')
plt.title('Age and department', fontsize = 20)
plt.xticks(np.arange(0, 95, step=10))
plt.show()

#%% Number of inhabitants per department

sns.lineplot(x= dataset1['DEP'], y= dataset1['inhabitants'], color = 'blue')
plt.title('Inhabitants and department', fontsize = 20)
plt.xticks(np.arange(0, 95, step=10))
plt.show()

#%% Proportion of the 4 most frequent degrees per departement

fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)

plt.subplot(221)
sns.lineplot(x= dataset1['DEP'],y= dataset1['D1_3'], color = 'blue')
plt.xticks(np.arange(0, 95, step=10))
       
plt.subplot(222)
sns.lineplot( x = dataset1['DEP'], y=dataset1['D1_6'], color = 'red')
plt.xticks(np.arange(0, 95, step=10))

plt.subplot(223)
sns.lineplot(x = dataset1['DEP'], y=dataset1['D1_7'], color = 'green')
plt.xticks(np.arange(0, 95, step=10))

plt.subplot(224)
sns.lineplot(x = dataset1['DEP'], y=dataset1['D1_8'], color = 'pink')
plt.xticks(np.arange(0, 95, step=10))

plt.show()

#%% Visualizing data from dataset 2

#%% Starting again with standard deviation

print(dataset2.std())

#%% Distribution of employee count and employer types

fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
plt.subplot(221)
plt.rcParams['figure.figsize'] = (15, 8)
sns.histplot(data=learn_test_complete,x="employee_count")
plt.subplot(222)
sns.histplot(data=learn_test_complete,x="Employer_type")

#%% Distribution of economic sectors

plt.rcParams['figure.figsize'] = (15, 8)
sns.histplot(data=learn_test_complete,x="economic_sector")
plt.xticks(rotation=70)

#%% Distribution of job condidtions, job categories

print(learn_test_complete["Job_condition"].value_counts())
print(learn_test_complete["JOB_CATEGORY"].value_counts())

#%% Distribution of types of contract

labels = ['APP', 'AUT', 'CDD', 'CDI', 'TOA', 'TTP']
size = learn_test_complete['TYPE_OF_CONTRACT'].value_counts()
colors = ['lightgreen', 'orange','blue','red','yellow','purple']
explode = [0.1, 0.1,0.1,0.1,0.1,0.1]

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%',pctdistance=1.2,labeldistance=1.3)
plt.title('Contract', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()

#%%  Distribution of working hours

sns.histplot(data=dataset2,x= "Working_hours")

#%% Mean working hours per department

sns.lineplot(x= dataset2['job_dep'], y= dataset2["Working_hours"], color = 'blue')

plt.title('Working hours and department', fontsize = 20)
plt.xticks(np.arange(0, 95, step=10))
plt.show()

#%% Mean emolument per department

sns.lineplot(x= dataset2['job_dep'], y= dataset2["Emolument"], color = 'blue')

plt.title('Emolument and department', fontsize = 20)
plt.xticks(np.arange(0, 95, step=10))
plt.show()

#%% Clustering algorithm

#%% Setting department numbers as indexes for clustering purposes

dataset1.set_index(["DEP"],inplace=True)
dataset2.set_index(["job_dep"],inplace=True)

#%% Clustering function

def Cluster(dataset,A,B): # The function takes one dataset and two of its columns as inputs
                          # It returns nothing but plots clusters ans dendrograms
    x=dataset[[A,B]].values # x is an array containing all values of the two given columns
    
    L = []
    for i in range(1, 11):
        km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1000, n_init = 100, random_state = 0)
        km.fit(x)
        L.append(km.inertia_)
        
    plt.plot(range(1, 11), L)
    plt.title('The Elbow Method', fontsize = 20)
    plt.xlabel('No. of Clusters')
    plt.ylabel('Within cluster Sum of Squares')
    plt.show()
    
    kn = KneeLocator(range(1, 11), L, curve='convex', direction='decreasing')
    print(kn.knee)    
    km = KMeans(n_clusters = kn.knee, init = 'k-means++', max_iter = 1000, n_init = 100, random_state = 0)
    km_predict = km.fit_predict(x)
    
    plt.scatter(x[:, 0], x[:, 1], c=km_predict, s=50, cmap='autumn')
    centers = km.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    
    plt.style.use('fivethirtyeight')
    plt.title('K Means Clustering', fontsize = 20)
    plt.xlabel(A)
    plt.ylabel(B)
    plt.grid()
    plt.show()
    
    dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

    plt.title('Dendrogam', fontsize = 20)
    plt.xlabel('DEP')
    plt.ylabel('Euclidean Distance')
    plt.show()

#%% Applying the algorithm to some couples of columns

Cluster(dataset1,"Female","age_2020")
#%%
Cluster(dataset1,"Female","inhabitants")
#%%
Cluster(dataset1,"D1_3","D1_8")
#%%
Cluster(dataset1,"D1_3","inhabitants")
#%%
Cluster(dataset1,"D1_3","TYPMR41")
#%%
Cluster(dataset2,"GZ","QA")
#%%
Cluster(dataset2,"tr_6","ct_9")
#%%
Cluster(dataset2,"CDD","ct_3")

#%% Dendrogram on the whole dataset1

x1=dataset1.values
dendrogram1 = sch.dendrogram(sch.linkage(x1, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)
plt.xlabel('DEP')
plt.ylabel('Euclidean Distance')
plt.show()

#%% Dendrogram on the whole dataset2

x2=dataset2.values
dendrogram2 = sch.dendrogram(sch.linkage(x2, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)
plt.xlabel('DEP')
plt.ylabel('Euclidean Distance')
plt.show()



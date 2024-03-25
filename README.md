Bike Sharing Demand Prediction - Capstone Project.ipynb_
Problem Description
Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.
Project Title : Seoul Bike Sharing Demand Prediction
Data Description
The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information.
Attribute Information:
Date : year-month-day
Rented Bike count - Count of bikes rented at each hour
Hour - Hour of he day
Temperature-Temperature in Celsius
Humidity - %
Windspeed - m/s
Visibility - 10m
Dew point temperature - Celsius
Solar radiation - MJ/m2
Rainfall - mm
Snowfall - cm
Seasons - Winter, Spring, Summer, Autumn
Holiday - Holiday/No holiday
Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)
[ ]
# Mount the google drive in google colab. 
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
[ ]
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10,6)
%matplotlib inline

# importing library called warning to ignore warnings.
import warnings
warnings.filterwarnings("ignore")
[ ]
#Loading dataset
df = pd.read_csv('/content/drive/MyDrive/Bike sharing demand prediction -Tanu Rajput/SeoulBikeData.csv',encoding= "ISO-8859-1")

[ ]
# peeking at the first five rows
df.head()
[ ]
# peeking at the last five rows
df.tail()
[ ]
# Data information
df.info()
OBSERVATION: Here the data type of the features, viz., Date, Seasons, Holiday, Functional Day is OBJECT

[ ]
# total number of Rows and Columns
df.shape
[ ]
#the statistical description of the features
df.describe().T
[ ]
df.describe(include='O').T
Preprocessing the Data
[ ]
# checking the null values
df.isna().sum()
[ ]
# checking duplicate values if any
df.duplicated().sum()
From the data information we saw that date is object datatype, so we have to convert it into date type

[ ]
# converting the datatype of date column from object to date.
df['Date']=pd.to_datetime(df['Date'])
Now we will create the separate date, month, year by extracting from the date column and then will drop the date column

[ ]
#extract day from date
df['WeekDay']=df["Date"].dt.day_name() 
#extract month from date
df['Month']=pd. DatetimeIndex(df['Date']).month_name()
#extract year from date
df['year']=df['Date'].dt.strftime('%Y')
[ ]
df.info()
[ ]
# Dropping the date column as we extracted all formats of date and keep them in separate columns respectively.
df.drop(columns=['Date'],inplace=True)
[ ]
# checking Outliers with seaborn boxplot
columnss = ['Rented Bike Count','Temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']
n = 1
plt.figure(figsize=(18,12))

for i in columnss:
  plt.subplot(3,3,n)
  n=n+1
  sns.boxplot(df[i])
  plt.title(i)
  plt.tight_layout()
We can see the outliers in rainfall and snowfall columns but we don't have to worry about outliers in this data, because if we treat the outliers from Rainfall and snowfall columns,it removes all the information of the data.

One more datatype is to be change

-->HOUR column datatype is of integer. But the date column is of timestamp and 'hour' is the part of timestamp. so it should be the categorical column

[ ]
# converting Hour column integer to Categorical 
df['Hour']=df['Hour'].astype('object')
EXPLORATORY DATA ANALYSIS & VISUALIZATION
[ ]
# setting the default fig size
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10,6)
[ ]
# Boxplots on Rental Bike Count
cat_columns=['Seasons','Hour','Holiday','WeekDay','year','Month']
n=1
plt.figure(figsize=(20,12))
for i in cat_columns:

  plt.subplot(3,3,n)
  n=n+1
  sns.boxplot(x=df[i],y=df['Rented Bike Count'])
  plt.title(f"Count over {i}")
  plt.tight_layout()
plt.show()
[ ]
# Seeing the distribution of target column = 'Rented Bike Count'
plt.figure(figsize=(8,6))
sns.histplot(df['Rented Bike Count'],bins=50, color = 'orange',)
plt.axvline(df['Rented Bike Count'].mean(), color='red', linestyle='dashed', linewidth=3)
plt.axvline(df['Rented Bike Count'].median(), color='indigo', linestyle='dotted', linewidth=3) 
plt.show()
sns.distplot(df['Rented Bike Count'])
plt.show()
Observation = The distribution of the Rented Bike Count is skewed to the right

[ ]
# normalizing the distribution of target column using square root
sns.distplot(np.sqrt(df['Rented Bike Count']),color='r')
plt.show()
[ ]
# seeing the data distribution of the numerical features
numcols=df[['Temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']]
n=1
for i in numcols:
  plt.subplot(9,3,n)
  plt.figure(figsize=(10,8))
  n+=1
  sns.distplot(df[i],color = 'indigo')
  plt.show()
Observation = As we can see, bike count is more in the year 2018 than 2017

[ ]
# total number of rented bike count per season
dfSeasons=pd.DataFrame(df.groupby('Seasons').sum()['Rented Bike Count'].sort_values(ascending=False))
dfSeasons
[ ]
dfSeasons.plot(kind='bar',color=['red','blue','pink','black'],y='Rented Bike Count')
plt.show()
Observation = Bike count in the summer season is more . While in winter the count is less

[ ]
#Rented Bike Count during Days of the Week
fig,ax=plt.subplots(figsize=(21,9))
sns.pointplot(x='Hour',y='Rented Bike Count',hue='WeekDay',data = df,ax=ax)
ax.set(title ='Rented Bike Count per hour during Days of the Week')
plt.show()
Observation: If we closely look into this pointplot, either its weekdays or weekend, the demand for rented bike count approx starts from morning 6 am. At 8am it is high and also from 6pm.
[ ]
# Rented bike count per day with respect to Month
fig,ax=plt.subplots(figsize=(21,9))
sns.pointplot(x='Hour',y='Rented Bike Count',hue='Month',data = df,ax=ax)
ax.set(title ='Rented Bike Count with respect to Month')
plt.show()
Observation: we get to know that in the month of December, January, February the demand for bike is less due to cold weather.
[ ]
# rented bike count per hour with respect to Seasons
fig,ax=plt.subplots(figsize=(21,9))
sns.pointplot(x='Hour',y='Rented Bike Count',hue='Seasons',data = df,ax=ax)
ax.set(title ='Rented Bike Count vs Seasons')
plt.show()
Observations: the demand for bike in summer is high and in winter is low.
[ ]
# sum percentage distribution of the rented bike count with respect to seasons
df.groupby('Seasons').sum()['Rented Bike Count'].plot.pie(autopct="%.2f%%")
plt.show()
[ ]
# average bike count with respect to weekdays
df.groupby('WeekDay')['Rented Bike Count'].mean().plot.barh(color=['#9A0EEA','indigo','blue','green','gold','orange','red'])
plt.show()
[ ]
# seeing the climate(sunlight) during the day
df.groupby('Hour').sum()['Solar Radiation (MJ/m2)'].plot(kind='bar', color='red',)
plt.show()
Observation: As we can see that the sunlight comes at 8am and it rises it peakes in the afternoon around 1pm and gradually decreases till 6pm. That's why people mostly used rented bike during these hours.
[ ]
# percentage distribution of the value counts of the categorical features
cols=['Month','Holiday','Seasons','year','WeekDay','Functioning Day']
n=1
plt.figure(figsize=(20,15))
for i in cols:
  plt.subplot(3,3,n)
  n=n+1
  plt.pie(df[i].value_counts(),labels = df[i].value_counts().keys().tolist(),autopct='%.0f%%')
  plt.title(i)
  plt.tight_layout()
[ ]
# Numerical feature analysis by ploting pair plot
sns.pairplot(df,corner=True,)
plt.show()
[ ]
# Regression Plots to know the relation between Target variable(rented bike count) and independent columns(Temperature, humidity, Solar radiation)

fig,(ax1,ax2,ax3)= plt.subplots(ncols=3, figsize = (22,5))
sns.regplot(df['Temperature(°C)'], df['Rented Bike Count'],scatter_kws={"color": "orange"}, line_kws={"color": "red"},ax=ax1)
ax1.set(title='Relation b/w Target variable and Temperature')
sns.regplot(df['Humidity(%)'], df['Rented Bike Count'],scatter_kws={"color": "blue"}, line_kws={"color": "red"},ax=ax2)
ax2.set(title='Relation b/w Target variable and Humidity')
sns.regplot(df['Solar Radiation (MJ/m2)'], df['Rented Bike Count'],scatter_kws={"color": "green"}, line_kws={"color": "red"},ax=ax3)
ax3.set(title='Relation b/w Target variable and Solar Radiation')
plt.show()
Correlation Matrix
[ ]
plt.figure(figsize=(13,10))
plt.title("Correlation Between Different Variables\n")
sns.heatmap(abs(df.corr()),
            cmap='hot', annot=True)     
plt.show()
Observation: We can see that there is strong correlation between the temperature and dew point temperature features which may cause trouble during the prediction. We will find/detect this type of multicollinearity in a different way ahead.
[ ]
# watching correlation between target variable and remaining independent variable
df.corr()['Rented Bike Count']
Observation: some features are negatively correlated and some positive with the target feature
Collinearity/Multicollinearity Detection
[ ]
# detecting multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor
attributes = df[['Temperature(°C)','Dew point temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']]
VIF = pd.DataFrame()
VIF["feature"] = attributes.columns 
#calculating VIF
VIF["Variance Inflation Factor"] = [variance_inflation_factor(attributes.values, i)
                          for i in range(len(attributes.columns))]
  
print(VIF)
Observation = As we can see that Temperature and Dew point temperature has high VIF.

Let's see the VIF after removing dew point temperature feature from the list.

[ ]
# VIF after removing the dew point temperature feature.

attributes = df[['Temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']]
VIF = pd.DataFrame()
VIF["feature"] = attributes.columns 
#calculating VIF
VIF["Variance Inflation Factor"] = [variance_inflation_factor(attributes.values, i)
                          for i in range(len(attributes.columns))]
print(VIF)
Observation = Now the VIF score is normal which is between 1 - 5.

Therefore we decided that it is better to remove the dew point temperature feature from the dataset.
[ ]
# dropping the dew point temperature feature
df=df.drop(['Dew point temperature(°C)'],axis=1)
Preparing for Data Modelling
Double-click (or enter) to edit

encoding process
[ ]
#encoding the categorical features.
final_df=pd.get_dummies(df,drop_first=True,sparse=True)
final_df.head(3).T
dividing the data
[ ]
# dividing the data into dependent variable(target) and independent variable
X = final_df.drop('Rented Bike Count',axis=1) # Independent features
y = np.sqrt(final_df['Rented Bike Count'])  # dependent features
train_test_split
[ ]
# importing required library
from sklearn.model_selection import train_test_split
[ ]
# train_test_split the data.
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25,random_state=42)
[ ]
X_train.head()
[ ]
X_test.head()
[ ]
X_train.shape, X_test.shape, y_train.shape, y_test.shape
Model Training
[ ]
# importing the models from sklearn library
from sklearn import model_selection
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor


# import evaluating metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt

# Import GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
Defining functions for finding metrics
[ ]
# Appending all models evalution scores to the corrosponding list after hyperarameter
MSE_ht=[]
RMSE_ht=[]
training_score_ht =[]
R2_ht=[]
ADJ_R2_ht=[]
[ ]
#defining a function for training the model and also the calculating the evaluation metrics
def eval_metric(model_name,X_train,X_test,y_train,y_test,linear = False):
  '''

    Defining the function to find the all evaluating metric scores

  '''  
  model_name.fit(X_train,y_train) #...fitting the model
  tr = model_name.score(X_train,y_train)#....to see the training set score
  print("Training_score =", tr)
  try:
    print("The best parameters is",model_name.best_params_)
  except:
    print('None')
  if linear == True:
    Y_pred = model_name.predict(X_test)
    mse  = mean_squared_error(y_test**2,Y_pred**2) #......... mean_squared_error
    print("MSE :" , mse)
    
    rmse = np.sqrt(mse) #..........root mean squared error
    print("RMSE :" ,rmse)
   
    r2 = r2_score(y_test**2,Y_pred**2)  #.......... r2 score
    print("R2 :" ,r2)
    
    adj_r2=1-(1-r2_score(y_test**2,Y_pred**2))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))  # ........adjusted r2 score
    print("Adjusted_R2 : ",adj_r2,'\n')
  else:
     
    Y_pred = model_name.predict(X_test)#.......for tree based models

    mse  = mean_squared_error(y_test,Y_pred)
    print("MSE :" , mse)

    rmse = np.sqrt(mse)
    print("RMSE :" ,rmse)
    
    r2 = r2_score(y_test,Y_pred)
    print("R2 :" ,r2)
   
    adj_r2=1-(1-r2_score(y_test,Y_pred))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
    print("Adjusted R2 : ",adj_r2,'\n')


  #appending all metrics for all models
  MSE_ht.append(mse)
  RMSE_ht.append(rmse)  
  R2_ht.append(r2)
  ADJ_R2_ht.append(adj_r2)
  training_score_ht.append(tr)

  if model_name == Lr:
    
    print('Coefficient:',model_name.coef_) #  ..... Coeff of linear model
    print('\n')
    print('Intercept:',model_name.intercept_) # ......intercept of linear model
  else:
    pass
  
while doing power transform,we should take care of some points::---

        1.No null values
        2.Not negative Value
        3.no 0
[ ]
# Feature transformation using Yeo Johnson transformation technique
from sklearn.preprocessing import PowerTransformer,MinMaxScaler
powertrans_ = PowerTransformer()
X_train_trans = powertrans_.fit_transform(X_train) #........ fit transform the training set
X_test_trans = powertrans_.transform(X_test)
All Models without hyperparameter tuning
[ ]
MSE=[]
RMSE=[]
training_score =[]
R2=[]
ADJ_R2=[]

# assigning models in variables
lr= LinearRegression()
l2 = Ridge()
l1 = Lasso()


linear_models = [lr,l1,l2]
for model_name in linear_models:
  model_name.fit(X_train_trans,y_train)
  y_pred = model_name.predict(X_test_trans)
  mse1 = mean_squared_error(y_test,y_pred)
  rmse1 = np.sqrt(mse1)
  r21 = r2_score(y_test,y_pred)
  ad_r21 =1-(1-r21)*((X_test_trans.shape[0]-1)/(X_test_trans.shape[0]-X_test_trans.shape[1]-1))

  training_score.append(model_name.score(X_train_trans,y_train))
  MSE.append(mse1)
  RMSE.append(rmse1)
  R2.append(r21)
  ADJ_R2.append(ad_r21)
[ ]
training_score
Linear Models with Hyperparameter Tuning
Linear Regression
[ ]
Lr =LinearRegression()
[ ]
# Fitting the linear regression model into defined function
eval_metric(Lr,X_train,X_test,y_train,y_test,linear = True)
Regularization = Lasso
[ ]
# using grid search CV for hyperparameter tuning of LASSO
parameters = {'alpha': [1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100,0.0014]} #lasso parameters 
L_lasso = GridSearchCV(Lasso(), parameters, cv=5) #using gridsearchcv and cross validate the model
[ ]
# fitting and calculating metric by calling the defined function
eval_metric(L_lasso,X_train,X_test,y_train,y_test,linear = True)
Regularizaion = Ridge
[ ]
# using grid search CV for hyperparameter tuning of RIDGE
parameters = {'alpha': [1e-15,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1,5,10,20,30,40,45,50,55,60,100,0.5,1.5,1.6,1.7,1.8,1.9]} # giving parameters 
L_ridge = GridSearchCV(Ridge(), parameters, scoring='r2', cv=5)
[ ]
# Fitting the Ridge model into the defined metric function
eval_metric(L_ridge,X_train,X_test,y_train,y_test,linear = True)
Polynomial
With polynomial degree 2
[ ]
polynomial_2 = PolynomialFeatures(2) #........creating variable with degree 2
poly_X_train2 = polynomial_2.fit_transform(X_train_trans) #........ fitting the train set
poly_X_test2 = polynomial_2.transform(X_test_trans) #.........transforming the test set
[ ]
# fitting and calculating metric by calling the defined function
eval_metric(Lr,poly_X_train2,poly_X_test2,y_train,y_test,linear = True)
Now, Tree Based Models without Hyperparameter
[ ]
# Feature scaling
scaling = MinMaxScaler()
[ ]
X_train_scaled  = scaling.fit_transform(X_train) #......fitting the X_train
X_test_scaled  = scaling.transform(X_test) # transform test set
[ ]
#assigning the models into new variables
d_tree= DecisionTreeRegressor()
r_forest = RandomForestRegressor()
g_boost = GradientBoostingRegressor()
xt_boost = ExtraTreesRegressor()

tree_models = [d_tree,r_forest,g_boost,xt_boost]
for model_name in tree_models:
  model_name.fit(X_train_scaled,y_train)
  y_pred = model_name.predict(X_test_scaled)
  mse = mean_squared_error(y_test,y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_test,y_pred)
  ad_r2 =1-(1-r2)*((X_test_scaled.shape[0]-1)/(X_test_scaled.shape[0]-X_test_scaled.shape[1]-1))
#appending the metrics into pre defined metric variables
  training_score.append(model_name.score(X_train_scaled,y_train))
  MSE.append(mse)
  RMSE.append(rmse)
  R2.append(r2)
  ADJ_R2.append(ad_r2)
Creating Dataframe of all metrics without hyperparameter tuning
[ ]
# all models into a list and storing in a new variable
models = ['Linear Regression','Ridge','Lasso','Decision_Tree','Random_Forest','Gradient_boost','ExtraTreeReg']

# Creating dictionary of evaluating metrics by creating new names
metrics = {'TRAININGSCORE':training_score,'MSE':MSE,'RMSE':RMSE,'R2':R2,'ADJ_R2':ADJ_R2}

# creating the dataframe of all metrics without hyperparameter tuning
metrics_df = pd.DataFrame.from_dict(metrics,orient='index',columns=models)

[ ]
metrics_df.T
Tree Models with Hyperparameter Tuning
DecisionTreeRegressor
[ ]
# Parameters for Decission Tree model
parameters = {'criterion':['mse'],#'squared_error', 'absolute_error',],
              'min_samples_leaf':[5],#7,10],
              'max_depth' : [18],#10,25],
              'min_samples_split': [25],#15,35],
              'max_features':['auto'],#'sqrt','log2']
              }
we put other parameters under comments as we found the best parameters among them.
[ ]
# using grid search CV
D_tree = GridSearchCV(DecisionTreeRegressor(),param_grid=parameters,cv=5,n_jobs=-1)
[ ]
# fitting and calculating metric by calling the defined function
eval_metric(D_tree,X_train_scaled,X_test_scaled,y_train,y_test)
RandomForestRegressor
[ ]
parameterss = {'n_estimators':[150],#100,200],
              'min_samples_leaf':[4],#6,2],
              'max_depth' : [20],#25,30],
              'min_samples_split': [25],#30,20],
              'max_features':['auto'],#'sqrt','log2']
              }
[ ]
# using grid search cv for hyperparameter
Random_forest_= GridSearchCV(RandomForestRegressor(),param_grid=parameterss,n_jobs=-1,cv=5)
[ ]
# fitting and calculating metric by calling the defined function
eval_metric(Random_forest_,X_train_scaled,X_test_scaled,y_train,y_test)
GradientBoostingRegressor
[ ]
parametersss={'loss':['huber'],#'squared_error', 'absolute_error','quantile'],
            'min_impurity_decrease':[0.4],#0.2,0.6],
            'criterion':['mse'],#'mae'],
            'n_estimators':[800],#600,400,1000], 
            'learning_rate': [0.01],#0.03,0.1,0.05], 
            'min_samples_leaf':[6],#4,8]
            'max_depth':[25],#15,20,30],
            'subsample':[0.7],#0.5,1.0],
            'max_leaf_nodes':[17],#15,10,20],
            'max_features':['auto']#'sqrt', 'log2'] 
            }
            
            
[ ]
gradient_boost_ = GridSearchCV(GradientBoostingRegressor(), param_grid=parametersss, n_jobs=-1,cv=5,verbose=2)
[ ]
# fitting and calculating metric by calling the defined function
eval_metric(gradient_boost_,X_train_scaled,X_test_scaled,y_train,y_test)
ExtraTreesRegressor
[ ]
param = {'n_estimators' : [100], 
         'max_depth' : [50],#60,70,80,90,100],
         'min_samples_split':[2],
         'min_samples_leaf':[1],
         'bootstrap' : [True],#False]
        }

# using grid search cv for hyperparameter
ExtraTrees_=GridSearchCV(ExtraTreesRegressor(),param_grid=param,n_jobs=-1,cv=5)
# fitting and calculating metric by calling the defined function
eval_metric(ExtraTrees_,X_train_scaled,X_test_scaled,y_train,y_test)
[ ]
# all models into a list and storing in a new variable
models_ht = ['Linear Regression','Ridge','Lasso','Polynomial','Decision_Tree','Random_Forest','Gradient_boost','ExtraTreesReg']
# Creating dictionary of evaluating metrics by creating new names
metrics_ht = {'TRAININGSCORE(ht)':training_score_ht,'MSE(ht)':MSE_ht,'RMSE(ht)':RMSE_ht,'R2(ht)':R2_ht,'ADJ_R2(ht)':ADJ_R2_ht}
# creating the dataframe of all metrics with hyperparameter tuning
metrics_df_ht = pd.DataFrame.from_dict(metrics_ht,orient='index',columns=models_ht)
[ ]
# sorting dataframe by adj_r2(ht)
T_ht = metrics_df_ht.T.sort_values('ADJ_R2(ht)',ascending=False)
[ ]
T_ht
Observation: 1) After hyperparameter tuning, we can consider the top three model but among them the best model is the Extra Trees regressor with a R2 score of 0.91573 and ADJ_R2 score of 0.909379
[ ]
# again training the ExtraTreesRegressor model to check the error between test data and predicted data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25,random_state=42)
[ ]
# fitting the extratreeregressor model
model_best = ExtraTreesRegressor()
model_best.fit(X_train,y_train)
y_pred = model_best.predict(X_test) 
[ ]
# visualizing the error
error = y_test - y_pred
fig,ax =plt.subplots()
ax.scatter(y_test,error,color='orange')
ax.axhline(lw=3,color='black')
plt.title("Error between predicted and test data using ExtraTreesRegressor model",fontsize=13)
plt.show()
Observation: we saw all model's error b/w test and predicted data. So, among all of them, extratreesregressor gives less error compare to others
CONCLUSION
[ ]
# fitting and calculating metric by calling the defined function
eval_metric(gradient_boost_,X_train_scaled,X_test_scaled,y_train,y_test)
1) We observed that bike rental count is high during week days then weekend days.
2) The rental bike counts are at its peak at 8 AM in the morning and 6pm in the evening.
3) We observed that people prefer to rent bikes during moderate to high temperature.
4) Highest rental bike count is during Autumn and summer seasons and the lowest in winter season.
5) Comparing the Adjusted R2 among all the models, ExtarTreesRegressor gives the highest Score where Adjusted R2 score is 0.908699 and Training score is 0.987167. Therefore, this model is the best for predicting the bike rental count on hour basis

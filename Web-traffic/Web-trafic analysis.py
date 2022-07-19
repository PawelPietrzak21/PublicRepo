import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import math

from sklearn.preprocessing import StandardScaler
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from scipy.stats import randint
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



# =============================================================================
# Visits dataset preparation
# =============================================================================

visits = pd.read_excel('exemplary_data.xlsx', sheet_name = 'visits')
visits.head()
visits.shape

visits['Time'] = pd.to_datetime(visits['godzina'], unit='h' ).dt.time
visits_selected = visits.loc[(visits['Data'] >='2014-11-10') & (visits['Data'] <='2014-11-23') ] 
visits_selected['Data'] = visits_selected['Data'].dt.date
visits_selected['Datetime'] = pd.to_datetime(visits_selected.Data.astype(str) +  ' ' + visits_selected.Time.astype(str))
visits_selected = visits_selected.set_index('Datetime')
visits_selected['DayOfWeek'] = visits_selected.index.dayofweek

sns.set_theme()

visits_selected['Wizyty_all'].plot(color = 'darkblue')
plt.title('Visits on website')
plt.xlabel('')
plt.show()



grp = visits_selected.groupby( ['godzina', 'DayOfWeek']).mean()['Wizyty_all']
grp.unstack().plot()
plt.ylabel('Visits')
plt.xlabel('Hour')
plt.title('Average daily visits to the website per day of week')
plt.show()



# =============================================================================
# Spots dataset preparation
# =============================================================================

spots = pd.read_excel('exemplary_data.xlsx', sheet_name = 'spots')
spots.head()
spots.shape



def convertTime(s):
        d = re.match(r'((?P<days>\d+) days, )?(?P<hours>\d+):'
                     r'(?P<minutes>\d+):(?P<seconds>\d+)', str(s)).groupdict(0)
        return dt.timedelta(**dict(((key, int(value)) for key, value in d.items())))
    
    
spots['TimeConv'] = pd.Series([convertTime(time) for time in spots['Time']])
spots['Datetime'] = spots['Date'] + spots['TimeConv']
spots = spots.set_index('Datetime')
spots.drop('TimeConv', axis=1, inplace=True)
spots['DayOfWeek'] = spots.index.dayofweek


# =============================================================================
# Datasets merge and columns selection
# =============================================================================


df = pd.merge_asof(spots, visits_selected, on='Datetime')
# df.drop(['Time_y', 'DayOfWeek_x', 'Data', 'Title', 'Date', 'Time_x', 'Commercial length', 'AGB Channel' ], axis=1, inplace=True)
df.rename(columns={'godzina':'hour'}, inplace=True)
df.rename(columns={'DayOfWeek_y':'DayOfWeek'}, inplace=True)
df.rename(columns={'Wizyty_all':'visits'}, inplace=True)
df['Day'] = pd.to_datetime(df.Datetime).dt.day

tmp = df['Datetime'].groupby([pd.to_datetime(df.Datetime).dt.day , pd.to_datetime(df.Datetime).dt.hour]).count()
tmp.index.names = ['Day','hour']
df = pd.merge(df,tmp, on=['Day','hour'])
df.rename(columns={'Datetime_y':'spotsPerHour'}, inplace=True)

df.drop(['Time_y', 'DayOfWeek_x', 'Data', 'Title', 'Date', 'Time_x', 'Commercial length', 'AGB Channel' ], axis=1, inplace=True)

# =============================================================================
# Exploratory Data Analysis
# =============================================================================

df.info()
df.isnull().sum()

plt.figure()
plt.title('Corelations of variables')
sns.heatmap(df.corr(), annot=True)
plt.show()

plt.figure(figsize=(8,10), facecolor='white')
sns.pairplot(df)
plt.show()

plt.figure()
sns.countplot(df['Campaign Channel'])
plt.xticks(rotation=90)
plt.show()

plt.figure()
sns.countplot(df['Timeband'])
plt.xticks(rotation=90)
plt.show()

plt.figure()
sns.countplot(df['BreakType (Block type)'])
plt.xticks(rotation=90)
plt.show()

plt.figure()
sns.countplot(df['Position Type in the block of commercials'])
plt.xticks(rotation=90)
plt.show()

plt.figure()
sns.histplot(df['spotsPerHour'])
plt.title('Histogram of the number of spots per hour')
plt.xticks(rotation=90)
plt.show()

plt.figure()
sns.histplot(df['visits'])
plt.xticks(rotation=90)
plt.show()

# =============================================================================
# Data Preprocessing
# =============================================================================

inputs = pd.get_dummies(df)
inputs.info()

X = inputs.drop(columns=['visits', 'Datetime_x'])
y= inputs['visits']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

features = X.columns
# vif =pd.DataFrame()
# vif['Features'] = X.columns
# vif['VIF'] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
# vif

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# =============================================================================
# Modeling
# =============================================================================


def evaluateModel(y_test, y_pred):
    print("*******************Results********************")
    print('The r2 score is:', r2_score(y_test, y_pred))
    print('The mean absolute error', mean_absolute_error(y_test, y_pred))
    print('The mean squared error', mean_squared_error(y_test, y_pred))
    print('root mean square error', math.sqrt(mean_squared_error(y_test, y_pred)))
    # cv = cross_val_score(model, X,y,cv=5)
    # print('The cross validation score', cv.mean())
    print("\n*****************XXXXXXXXXXX********************")

# Linear Regression

lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

evaluateModel(y_test, y_pred)

par_grid =  {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid_lm = GridSearchCV(estimator=lm, param_grid=par_grid, cv=5,n_jobs=1, verbose=1)
grid_lm.fit(X_train, y_train)
y_pred1 = grid_lm.predict(X_test)

print("The best score:", grid_lm.best_score_)

evaluateModel(y_test, y_pred1)


# Lasso Regularization

lasso = Lasso()
lasso.fit(X_train,y_train)

y_pred = lasso.predict(X_test)

evaluateModel(y_test, y_pred)


param = {'alpha': np.arange(0.0001,0.1,0.001)}
grid_lass= GridSearchCV(estimator=lasso,param_grid=param,n_jobs=2,cv=5,verbose=2)

grid_lass.fit(X_train,y_train)
y_pred1 = grid_lass.predict(X_test)

print("The best score:", grid_lass.best_score_)

evaluateModel(y_test, y_pred1)

# KNN
print('KNN')

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train,y_train)
y_pred = knn_reg.predict(X_test)

evaluateModel(y_test, y_pred)

param = {'algorithm':['kd_tree'], 
         'n_neighbors':[3,2,4,6,8,10,14,7,11]}
grid_knn = GridSearchCV(estimator=knn_reg, param_grid=param)
grid_knn.fit(X_train,y_train)

grid_knn.fit(X_train,y_train)

y_pred1 = grid_knn.predict(X_test)

print("The best score:", grid_knn.best_score_)

evaluateModel(y_test, y_pred1)

# Random Forest
print('Random Forest')
random_reg = RandomForestRegressor()
random_reg.fit(X_train,y_train)
y_pred = random_reg.predict(X_test)

evaluateModel(y_test, y_pred)


n_estimator = [int(x) for x in np.linspace(start=10, stop=120,num=10)]
max_features = ['auto','sqrt']
max_depth=[int(x) for x in np.linspace(5,50,num=6)]
min_samples_split = [2,5,3,7,8,4]
min_samples_leaf = [1,3,2,5,7,8,4,12,15,17,9,20]

param = {'n_estimators':n_estimator, 'max_features':max_features, 'max_depth':max_depth, 'min_samples_leaf':min_samples_leaf, 'min_samples_split':min_samples_split}

random_cv = RandomizedSearchCV(estimator=random_reg, param_distributions=param, n_iter=4, cv=5, n_jobs=2,verbose=2)
random_cv.fit(X_train, y_train)
y_pred1 = random_cv.predict(X_test)

print("The best score:", random_cv.best_score_)

evaluateModel(y_test, y_pred1)

# Gradient boosting
print('Gradient boosting')
gb_reg = GradientBoostingRegressor()
gb_reg.fit(X_train,y_train)
y_pred = gb_reg.predict(X_test)

evaluateModel(y_test, y_pred)

params = {"n_estimators":[50,100,200,300,400,500,600,700,800,900],"max_depth":[3,4,5,6,7,8,9,10,12,15],"min_samples_split":[2,5,8,10,12,15,18,20,22],
             "max_features":['auto','sqrt'],"min_samples_leaf":[1,3,5,6,7,8],"learning_rate":[0.01,0.05,0.1,0.3,0.5,0.6,0.7]}

random_gb = RandomizedSearchCV(gb_reg,param_distributions=params,n_iter=30,n_jobs=2,cv=6,verbose=2)
random_gb.fit(X_train,y_train)
          
y_pred1 = random_gb.predict(X_test)

print("The best score:", random_gb.best_score_)

evaluateModel(y_test, y_pred1)


# Decision tree
print('Decision tree')

tree_reg = DecisionTreeRegressor(random_state = 0) 
tree_reg.fit(X_train,y_train)
y_pred = tree_reg.predict(X_test)

evaluateModel(y_test, y_pred)


parameters={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8],
           "min_weight_fraction_leaf":[0.1,0.25,0.5,0.75,0.9],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,25,50,75,90] }

tree_cv = GridSearchCV(tree_reg,param_grid=parameters,cv=3,verbose=3)
tree_cv.fit(X_train, y_train)

y_pred1 = tree_cv.predict(X_test)
# tree_cv.best_params()
print("The best score:", tree_cv.best_score_)

evaluateModel(y_test, y_pred1)


# =============================================================================
# Selected model - tree_reg
# =============================================================================

from sklearn.tree import export_graphviz 


export_graphviz(tree_reg, out_file ='tree.dot', feature_names = features)


text_representation = tree.export_text(tree_reg)
print(text_representation)
with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)



fig = plt.figure(figsize=(50,40))
_ = tree.plot_tree(tree_reg, 
                   feature_names=features,  
                   class_names='visits',
                   filled=True)
fig.savefig("decistion_tree.png")


feat_importances = pd.Series(tree_reg.feature_importances_, index=features)
plt.title('Features importance')
feat_importances.nlargest(20).plot(kind='barh')


tree_reg.get_depth()
tree_reg.get_n_leaves()
tree_reg.decision_path(X_test)



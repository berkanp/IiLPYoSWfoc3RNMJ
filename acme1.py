
import pandas as pd
import statsmodels.api as sm
import seaborn
seaborn.set()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix



data = pd.read_csv('ACME-HappinessSurvey2020.csv')
data.describe()


# g = sns.FacetGrid(data, col='Y')
# g.map(plt.hist, 'X1', bins=20)

# g = sns.FacetGrid(data, col='Y')
# g.map(plt.hist, 'X2', bins=20)

# g = sns.FacetGrid(data, col='Y')
# g.map(plt.hist, 'X3', bins=20)

# g = sns.FacetGrid(data, col='Y')
# g.map(plt.hist, 'X4', bins=20)

# g = sns.FacetGrid(data, col='Y')
# g.map(plt.hist, 'X5', bins=20)

# g = sns.FacetGrid(data, col='Y')
# g.map(plt.hist, 'X6', bins=20)


data=[data]
for dataset in data:    
    dataset.loc[ dataset['X1'] == 1, 'X1'] = 0
    dataset.loc[ (dataset['X1'] >= 2) & (dataset['X1'] < 4), 'X1'] = 1
    dataset.loc[ dataset['X1'] > 4, 'X1'] = 2
for dataset in data:    
    dataset.loc[ dataset['X2'] == 1, 'X2'] = 0
    dataset.loc[ (dataset['X2'] >= 2) & (dataset['X2'] < 4), 'X2'] = 1
    dataset.loc[ dataset['X2'] > 4, 'X2'] = 2
for dataset in data:    
    dataset.loc[ dataset['X3'] == 1, 'X3'] = 0
    dataset.loc[ (dataset['X3'] >= 2) & (dataset['X3'] < 4), 'X3'] = 1
    dataset.loc[ dataset['X3'] > 4, 'X3'] = 2
for dataset in data:    
    dataset.loc[ dataset['X4'] == 1, 'X4'] = 0
    dataset.loc[ (dataset['X4'] >= 2) & (dataset['X4'] < 4), 'X4'] = 1
    dataset.loc[ dataset['X4'] > 4, 'X4'] = 2
for dataset in data:    
    dataset.loc[ dataset['X5'] == 1, 'X5'] = 0
    dataset.loc[ (dataset['X5'] >= 2) & (dataset['X5'] < 4), 'X5'] = 1
    dataset.loc[ dataset['X5'] > 4, 'X5'] = 2
for dataset in data:    
    dataset.loc[ dataset['X6'] == 1, 'X6'] = 0
    dataset.loc[ (dataset['X6'] >= 2) & (dataset['X6'] < 4), 'X6'] = 1
    dataset.loc[ dataset['X6'] > 4, 'X6'] = 2


#p value higher than 0.5 : X2 and X4
x=dataset.drop(['Y','X2','X4'], axis=1)
y=dataset['Y']





results = sm.OLS(y,x).fit()
results.summary()


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.10, random_state=1)


import lightgbm as ltb

lgb = ltb.LGBMClassifier(n_estimators=80, 
                          colsample_bytree=0.7,  
                          min_child_weight=1, 
                          max_depth=9, 
                          subsample=0.65).fit(x_train, y_train)

y_pred = lgb.predict(x_test)

print('Accuracy of LGBMClassifier: ')
acc= round (accuracy_score(y_test, y_pred)*100,2)
print(acc)
acc_rfc =round(lgb.score(x_train, y_train) * 100, 2) 
print(acc_rfc)
cm = confusion_matrix(y_test,y_pred)
print(cm)

# lgb =ltb.LGBMClassifier()
# lgb_params = {

#                 'n_estimator':[80,79,81],

#                 'subsample':[0.65,0.64]

#               }
# lgb_model = GridSearchCV(lgb, lgb_params,cv=10, n_jobs=-1, verbose=2).fit(x_train,y_train)
# lgb_model.best_params_
# print(lgb_model.best_params_)

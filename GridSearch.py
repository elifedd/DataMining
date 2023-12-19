from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn. metrics import explained_variance_score, mean_absolute_error, r2_score

train,test = train_test_split(data,test_size = 0.25,random_state = 0 )
train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']

model_ridge = Ridge()
parameters = {'learning_rate': [0.01,0.02,0.03,0.04],
                  'subsample'    : [0.9, 0.5, 0.2, 0.1],
                  'n_estimators' : [100,500,1000, 1500],
                  'max_depth'    : [4,6,8,10]
                 }
parameters_ridge = {
    'solver':['svd', 'cholesky', 'lsqr', 'sag'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
    'fit_intercept':[True, False]
    }

grid_ridge = GridSearchCV (model_ridge, parameters_ridge, scoring='neg_mean_absolute_error', n_jobs=-1,cv = 2)
grid_ridge.fit(train_X, train_Y)

print(" Results from Grid Search for Ridge " )
print("\n The best estimator across ALL searched params:\n",grid_ridge.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_ridge.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_ridge.best_params_)

from time import time

regressors = [
    KNeighborsRegressor(),
    GradientBoostingRegressor(),

    ExtraTreesRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    LinearRegression(),
    Lasso(),
    Ridge()
]
head = 10
for model in regressors[:head]:
    start = time()
    model.fit(train_X, train_Y)
    train_time = time() - start
    start = time()
    y_pred = model.predict(test_X)
    predict_time = time() - start
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print("\tExplained variance:", explained_variance_score(test_Y, y_pred))
    print("\tMean absolute error:", mean_absolute_error(test_Y, y_pred))
    print("\tR2 score:", r2_score(test_Y, y_pred))
    print()
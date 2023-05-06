import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, RandomForestRegressor, 
                                BaggingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import multiprocessing as mp

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
Y = pd.Series(diabetes.target, name='response')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


class model_init():
    modelmlg = LinearRegression()
    modeldcr = DecisionTreeRegressor(random_state=42)
    modelbag = BaggingRegressor(random_state=42)
    modelrfr = RandomForestRegressor(random_state=42)
    modelXGR = xgb.XGBRegressor()
    modelKNN = KNeighborsRegressor(n_neighbors=5)
    modelETR = ExtraTreesRegressor(random_state=42)
    modelRE=Ridge(random_state=42)
    modelLO=linear_model.Lasso(random_state=42,alpha=0.1)
    
    modelGBR = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                          criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                          min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                          init=None, random_state=None, max_features=None,
                                          alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
                                          validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)

mi_grid = model_init()
linear_regression = {
   'LinearRegression':mi_grid.modelmlg,
}
decision_tree_regressor = {
   'DecisionTreeRegressor':mi_grid.modeldcr,
   "splitter": ["best"],
    "criterion": ["squared_error", "absolute_error"],
    "min_samples_split": [10, 20],
    "max_depth": [2, 6],
    "min_samples_leaf": [20, 40],
    "max_leaf_nodes": [5, 20],
    "max_features": [1,2,4,6]
}

models_dict_grid={'LinearRegression':mi_grid.modelmlg,
                  'DecisionTreeRegressor':mi_grid.modeldcr,
                  'RandomForestRegressor':mi_grid.modelrfr,
                    'KNeighborsRegressor':mi_grid.modelKNN,
                    'ExtraTreesRegressor':mi_grid.modelETR,
                    'GradientBoostingRegressor':mi_grid.modelGBR,
                    'XGBRegressor':mi_grid.modelXGR,
                    'BaggingRegressor':mi_grid.modelbag,
                    'Ridge Regression':mi_grid.modelRE,
                    'Lasso Regression':mi_grid.modelLO}

models_dict_grid_params = {
       'LinearRegression':{},
       'DecisionTreeRegressor':{
                                # "splitter": ["best", "random"],
                                # "criterion": ["squared_error", "absolute_error"],
                                # "min_samples_split": [10, 20, 40],
                                # "max_depth": [2, 6, 8],
                                # "min_samples_leaf": [20, 40, 100],
                                # "max_leaf_nodes": [5, 20, 100],
                                # "max_features": ["auto","sqrt","log2"]
                                "splitter": ["best"],
                                "criterion": ["squared_error", "absolute_error"],
                                "min_samples_split": [10, 20],
                                "max_depth": [2, 6],
                                "min_samples_leaf": [20, 40],
                                "max_leaf_nodes": [5, 20],
                                "max_features": [1,2,4,6]
                                },
       'RandomForestRegressor':{ 
                                # 'bootstrap': [True],
                                # 'max_features': ['auto','log2'],
                                # 'n_estimators': [100,120,180], 
                                # 'max_depth': [None, 1, 2, 3], 
                                # 'min_samples_split': [2, 4, 6]},
                                'bootstrap': [True],
                                "max_features": [1,2,4,6],
                                'n_estimators': [100,120], 
                                'max_depth': [None, 1, 2, 3], 
                                'min_samples_split': [2, 4]},
        'KNeighborsRegressor':{
                              #   'n_neighbors': [2,3,4,5,6,7,], 
                              #  'weights': ['uniform','distance'],
                              #  'p':[1,2,5]},
                                'n_neighbors': [2,3], 
                               'weights': ['uniform','distance'],
                               'p':[1,2]},
        'ExtraTreesRegressor':{
                                # 'n_estimators': [int(x) for x in np.arange(start = 100, stop = 300, step = 100)],
                                # 'criterion': ['squared_error', 'absolute_error'],
                                # 'max_depth': [2,8,16,32,50],
                                # 'min_samples_split': [2,4,6],
                                # 'min_samples_leaf': [1,2],
                                #  'bootstrap': [True, False],
                                # 'warm_start': [True, False],
                                'n_estimators': [int(x) for x in np.arange(start = 100, stop = 110, step = 5)],
                                'criterion': ['squared_error', 'absolute_error'],
                                'max_depth': [2,8],
                                'min_samples_split': [2,4],
                                'min_samples_leaf': [1,2],
                                 'bootstrap': [True, False],
                                'warm_start': [True, False],
                                'max_features':[1,2,4,6]
                             },
        'GradientBoostingRegressor':{
                                      # 'learning_rate': [0.01,0.02,0.03,0.04],
                                      # 'subsample'    : [0.9, 0.5, 0.2, 0.1],
                                      # 'n_estimators' : [10,20,30,40],
                                      # 'max_depth'    : [4,6,8,10]
                                      'learning_rate': [0.01,0.02],
                                      'subsample'    : [0.9, 0.5],
                                      'n_estimators' : [10,20],
                                      'max_depth'    : [4,6],
                                      'max_features':[1,2,4,6]
                                    },
        'XGBRegressor':{
                        # 'learning_rate': [.03, 0.05, .07], #so called `eta` value
                        # 'max_depth': [5, 6, 7],
                        # 'min_child_weight': [4],
                        # "gamma":[ 0.0, 0.1, 0.2],
                        # 'subsample': [0.7],
                        # 'colsample_bytree': [0.7],
                        # 'n_estimators': [10]
                        'learning_rate': [.03, 0.05], #so called `eta` value
                        'max_depth': [5, 6],
                        'min_child_weight': [4],
                        "gamma":[ 0.0, 0.1],
                        'subsample': [0.7],
                        'colsample_bytree': [0.7],
                        'n_estimators': [10]
                        },
        'BaggingRegressor':{
                            # 'n_estimators': [10,15,20],
                            # 'max_samples' : [0.05, 0.1, 0.2, 0.5],
                            # 'max_features': [1,2,3,4,5],
                            # 'bootstrap': [True, False]
                            'n_estimators': [10,15],
                            'max_samples' : [0.05, 0.1],
                            'max_features': [1,2],
                            'bootstrap': [True, False]
                        },
        'Ridge Regression':
                          {
                            # 'alpha':[1, 10]
                            }
                          ,
        'Lasso Regression':{
                              # 'alpha': np.arange(0.00, 1.0, 0.1)
                              'alpha': np.arange(0.00, 1.0, 0.1)
                              }
    }

# [(Model, Params)]
items = [(X_train, Y_train, X_test, Y_test, 'LinearRegression',
          models_dict_grid['LinearRegression'],models_dict_grid_params['LinearRegression']),
         (X_train, Y_train, X_test, Y_test,'DecisionTreeRegressor',
          models_dict_grid['DecisionTreeRegressor'], models_dict_grid_params['DecisionTreeRegressor']),
         (X_train, Y_train, X_test, Y_test,'RandomForestRegressor',
          models_dict_grid['RandomForestRegressor'], models_dict_grid_params['RandomForestRegressor']),
         (X_train, Y_train, X_test, Y_test,'KNeighborsRegressor',
          models_dict_grid['KNeighborsRegressor'], models_dict_grid_params['KNeighborsRegressor']),
         (X_train, Y_train, X_test, Y_test,'ExtraTreesRegressor',
          models_dict_grid['ExtraTreesRegressor'], models_dict_grid_params['ExtraTreesRegressor']),
         (X_train, Y_train, X_test, Y_test,'GradientBoostingRegressor',
          models_dict_grid['GradientBoostingRegressor'], models_dict_grid_params['GradientBoostingRegressor']),
         (X_train, Y_train, X_test, Y_test,'XGBRegressor',
          models_dict_grid['XGBRegressor'], models_dict_grid_params['XGBRegressor']),
         (X_train, Y_train, X_test, Y_test,'BaggingRegressor',
          models_dict_grid['BaggingRegressor'], models_dict_grid_params['BaggingRegressor']),
         (X_train, Y_train, X_test, Y_test,'Ridge Regression',
          models_dict_grid['Ridge Regression'], models_dict_grid_params['Ridge Regression']),
         (X_train, Y_train, X_test, Y_test,'Lasso Regression',
          models_dict_grid['Lasso Regression'], models_dict_grid_params['Lasso Regression'])]


grid_dict ={'Model Name':[], 'R2_score':[], 'Root_Mean_Squared_Error_RMSE':[],  'Mean_Squared_Error_MSE':[]}
grid_df=pd.DataFrame(grid_dict)   


def grid_results(X_train, Y_train, X_test, Y_test, model_name ,model, params):
    grid_dict ={'Model Name':[], 'R2_score':[], 'Root_Mean_Squared_Error_RMSE':[],  'Mean_Squared_Error_MSE':[]}
    grid_df=pd.DataFrame(grid_dict)    
    

    grid_store={} # store the grid search results here

    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid.fit(X_train, Y_train)

    Y_pred_test = grid.predict(X_test)
    new_row = {'Model Name' : model_name,
            'R2_score' : r2_score(Y_test, Y_pred_test),
            'Root_Mean_Squared_Error_RMSE' : np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test)),
            'Mean_Squared_Error_MSE' : mean_squared_error(Y_test, Y_pred_test),
            'best_max_features' : grid.best_params_,
            'optimal_n_estimators':grid.best_score_}
    print(f"\n{model_name} complete\n")
      
    if model_name not in grid_store.keys():
        grid_store[model_name] = [grid.best_params_, grid.best_score_]
    
    return grid_store

if __name__ == '__main__':
    import time
    start = time.time()
    with mp.Pool() as pool:
        for result in pool.starmap(grid_results, items):
            print(result)
    end = time.time()
    print(f"Total runtime: {end-start}s")
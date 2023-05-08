import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, RandomForestRegressor, 
                                BaggingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pickle
import os

class RegressionModelSelector():
    """
    This class allows for the appropriate model to be selected from a standard set of 10
    Models are regression based
    """
    def __init__(self, df):
        self.df = df

    def feature_creation(self):
        """
        Create the month and year features to separate them out from YYYY-MM
        YYYY and mm are valuable features in their own right
        """
        df = self.df
        df["month"] = pd.to_datetime(df['Month'], format="%Y-%m").dt.month
        df["year"] = pd.to_datetime(df['Month'], format="%Y-%m").dt.year
        df.drop(["Month"], axis = 1, inplace = True)
        self.df = df
    
    def train_test_split(self):
        """
        Create the train, validatation and test set
        
        Stores the following in memory for ease of use:
            X_train (list): Training set
            X_val (list): Validation set
            X_test (list): Test set
            y_train (list): Train set prediction
            y_val (list): Validation set prediction
            y_test (list): Testing set prediction
        """
        df = self.df
        X = df.loc[:, ['Term', 'month', 'year']]
        y = df.loc[:,'avg_price']
        X_train_or, X_test, y_train_or, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train_or, y_train_or, test_size=0.25)
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train 
        self.y_val = y_val
        self.y_test = y_test
    
    def set_models(self):
        """
        Define the models to be trained as well as the dictionary to loop over
        Store the models into a dictionary of <model name> : <model object>
        """
        modelmlg = LinearRegression()
        modeldcr = DecisionTreeRegressor()
        modelbag = BaggingRegressor()
        modelrfr = RandomForestRegressor()
        modelXGR = xgb.XGBRegressor()
        modelKNN = KNeighborsRegressor(n_neighbors=5)
        modelETR = ExtraTreesRegressor()
        modelRE=Ridge()
        modelLO=linear_model.Lasso(alpha=0.1)
        
        modelGBR = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                             criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                             min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                             init=None, random_state=None, max_features=None,
                                             alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
                                             validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)

        # Evalution matrix for all the algorithms

        # MM = [modelmlg, modeldcr, modelrfr, modelKNN, modelETR, modelGBR, modelXGR, modelbag,modelRE,modelLO]
        models_dict={'LinearRegression':modelmlg,'DecisionTreeRegressor':modeldcr,'RandomForestRegressor':modelrfr,
                'KNeighborsRegressor':modelKNN,'ExtraTreesRegressor':modelETR,'GradientBoostingRegressor':modelGBR,
                'XGBRegressor':modelXGR,'BaggingRegressor':modelbag,'Ridge Regression':modelRE,'Lasso Regression':modelLO}
        
        self.models_dict = models_dict
    
    def MAPE (self, y_test, y_pred):
        """
        Mean Absolute Percentage Error

        Args:
            y_test (series): Data for predicting test on
            y_pred (series): Predicted outcomes
        
        Returns:
            (float): The mean absolute percentage error
        """
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    def fit_model(self, test_param: str):
        """
        Iterate through a defined set of models and add them to a dictionary

        Args:
            test_param (str): Either validation or test
            
        Returns:
            results (pd.DataFrame): Dataframe with results from each model per row 
        
        """
        results_dict ={'Model Name':[], 'Mean_Absolute_Error_MAE':[] ,
                   'Root_Mean_Squared_Error_RMSE':[] ,'Mean_Absolute_Percentage_Error_MAPE':[] ,
                   'Mean_Squared_Error_MSE':[] ,'Root_Mean_Squared_Log_Error_RMSLE':[] ,'R2_score':[]}
        results=pd.DataFrame(results_dict)
        
        if test_param == "validation":
            X_2 = self.X_val
            y_2 = self.y_val
        elif test_param == "test":
            X_2 = self.X_test
            y_2 = self.y_test
        for name, models in self.models_dict.items():
            models.fit(self.X_train, self.y_train)
            y_pred = models.predict(X_2)

            # Metrics
            result = self.MAPE(y_2, y_pred)    
            new_row = {'Model Name' : name,
                       'Mean_Absolute_Error_MAE' : metrics.mean_absolute_error(y_2, y_pred),
                       'Root_Mean_Squared_Error_RMSE' : np.sqrt(metrics.mean_squared_error(y_2, y_pred)),
                       'Mean_Absolute_Percentage_Error_MAPE' : result,
                       'Mean_Squared_Error_MSE' : metrics.mean_squared_error(y_2, y_pred),
                       'Root_Mean_Squared_Log_Error_RMSLE': np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_2)))),
                       'R2_score' : metrics.r2_score(y_2, y_pred)}
            results = results.append(new_row, ignore_index=True)
            return results
    
    def model_scores(self, results):
        """
        Print model scores for the end user
        End user can then select which model they want to save
        """
        print(results.iloc[results['Mean_Absolute_Error_MAE'].idxmin()][['Model Name',"Mean_Absolute_Error_MAE"]], 
              "\n\n",
                results.iloc[results['Root_Mean_Squared_Error_RMSE'].idxmin()][['Model Name',"Root_Mean_Squared_Error_RMSE"]], 
              "\n\n",
                results.iloc[results['Mean_Absolute_Percentage_Error_MAPE'].idxmin()][['Model Name',"Mean_Absolute_Percentage_Error_MAPE"]], 
              "\n\n",
                results.iloc[results['Mean_Squared_Error_MSE'].idxmin()][['Model Name',"Mean_Squared_Error_MSE"]],"\n\n",
                results.iloc[results['R2_score'].idxmax()][['Model Name',"R2_score"]],
                "\n\n")
    
    def auto_run(self):
        """
        Automatically run everyting
        This includes feature engineering, splitting up data for train, validation and test
        Finally, print model scores in CLI for user to pick
        """
        self.feature_creation()
        self.train_test_split()
        self.set_models()
        results = self.fit_model("test")
        
        self.model_scores(results)

        self.save_model()
    
    def save_model(self):
        """_
        Save or don't save a selected model
        """
        save_model = input("Save model? (Y/N): ")
        save_model = save_model.lower()
        if save_model == "y":
            chosen_model = input("Chosen model name: ")
            my_model = self.models_dict[chosen_model]
            file_name = os.path.join(os.getcwd(), "models", chosen_model + ".pkl")

            # save
            pickle.dump(my_model, open(file_name, "wb"))
            print(f"Model saved to {os.path.join(os.getcwd(), 'models')}")
        elif save_model == "n":
            print("No model selected")
            return
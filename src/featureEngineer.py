import os
import sys
import pandas as pd
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, Normalizer,
                                   PolynomialFeatures, LabelEncoder, OneHotEncoder)
import numpy as np

class FeatureEngineerPD():
    """
    Class to perform all the necessary feature engineering as needed

    References:
        https://towardsdatascience.com/fast-feature-engineering-in-python-tabular-data-d050b68bb178
    """
    def __init__(self, df: pd.DataFrame) -> pd.Dataframe:
        self.df = df

    def split_x_y(self, df:pd.DataFrame, col_x: list, col_y: list) -> pd.DataFrame:
        """
        Split the input dataframe into the X and Y dataframes.
        Split is based on user input for what is decided to be X and Y

        Args:
            df (pd.DataFrame): Input dataframe
            col_x (list): List of column names for X
            col_y (list): List of column names for Y

        Returns:
            (pd.DataFrame, pd.DataFrame) (tuple): Tuple containing the X and Y dataframes
        """
        X_df = df.loc[:,col_x]
        Y_df = df.loc[:,col_y]

        return (X_df, Y_df)

    def scale(self, df: pd.DataFrame, feature_range=(-1, 1)) -> pd.DataFrame:
        """
        Scaler using sklearn.preprocessing.MinMaxScaler
        Used for numerical data

        Args:
            df (pd.DataFrame): Input dataframe
            feature_range (tuple): Feature range. (-1,1) default, or (0, 1)
                                    Depends on whether feature has positive values or not
        
        Returns:
            df_scaled (pd.DataFrame): Scaled dataframe based on input feature range
        """
        scaler = MinMaxScaler(feature_range=feature_range)
        df_scaled = scaler.fit_transform(df)
        return df_scaled

    def standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizer using sklearn.preprocessing.StandardScaler
        Used for numerical data

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            df_standard (pd.DataFrame): Standardized dataframe
        """
        standard_scaler = StandardScaler()
        df_standard = standard_scaler.fit_transform(df)
        return df_standard

    def normal(self, df: pd.DataFrame, norm:str = "l2") -> pd.DataFrame:
        """
        Normalizer sklearn.preprocessing.Normalizer
        Used for numerical data

        Args:
            df (pd.DataFrame): Input dataframe
            norm (str): Normalizer method. Default l2 regularization.

        Returns:
            df_normal (pd.DataFrame): Normalized dataframe
        """
        normalizer = Normalizer(norm = norm)
        df_normal = normalizer.fit_transform(df)
        return df_normal

    def poly_feature(self, df: pd.DataFrame, degree=2, bias=False) -> pd.DataFrame:
        """
        Create polynomial features
        Used for numerical data

        Args:
            df (pd.DataFrame): Input dataframe
            degrees (int, optional): Degress for the polynomial. Defaults to 2.
            bias (bool, optional): _description_. Defaults to False.

        Returns:
            df_poly (pd.DataFrame): Polynomial features dataframe
        """
        print("""Building polynomial features will lead to feature explosion so be sure to use good feature selection practices.""")
        poly = PolynomialFeatures(degree=degree, include_bias=bias)
        df_poly = poly.fit_transform(df)
        return df_poly
    
    def bin(self, df: pd.DataFrame, cols_list:list ,bins:list) -> pd.DataFrame:
        """
        Binning method to treat numerical variables as categorical
        Used for numerical data

        Args:
            df (pd.DataFrame): Input dataframe
            cols_list (list): List of columns to bin
            bins (list): List of values to bin data into discrete bins

        Returns:
            df (pd.DataFrame): Dataframe with binned values set as new column
        """
        # Could be optimized here... for later thought
        for name in cols_list:
            df[name+"_binned"] = pd.cut(df[name], bins=bins)
    
        return df
    
    def label_encode(self, df: pd.DataFrame, category_cols:list) -> pd.DataFrame:
        """
        Encode all categorical features as a number
        Label Encoding is the encoding of choice when we are encoding ordinal variables
        Ensure that encodings are corresponding with order of categories
        
        Args:
            df (pd.DataFrame): Input dataframe
            category_cols (list): List of columns that are categorical

        Returns:
            df (pd.DataFrame): Dataframe with new columns that have the label encoded variables
        """
        label_encoder = LabelEncoder()
        
        for cat in category_cols:
            df.loc[:, cat] = label_encoder.fit_transform(df[cat])

        return df

    def onehot_encod(self, df: pd.DataFrame, category_cols:list, sparse:bool = True) -> pd.DataFrame:
        """
        One Hot Encoding is a technique where we encode each category as a vector of binary values where 
            the length of the vector is the number of unique categories in our feature

        Suitable for regression models
        
        Args:
            df (pd.DataFrame): Input dataframe
            category_cols (list): List of columns that are categorical
            sparse (bool): Calibrate output of encoder. Default set to True to save memory.

        Returns:
            df (pd.DataFrame): Dataframe with new columns that have the label encoded variables
        """
        oh_encoder = OneHotEncoder(sparse=sparse)
        for cat in category_cols:
            df.loc[:, cat] = oh_encoder.fit_transform(df[cat].values.reshape(-1, 1))
        
    def cat_to_num(self, df: pd.DataFrame, category_cols:list, single:bool=True) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): Input dataframe
            category_cols (list): List of columns that are categorical
            single (bool): Whether to set a single column or multiple from cat to num. Default set to True.

        Returns:
            pd.DataFrame: _description_
        """
        if single:
            for cat in category_cols:
                df[cat+"_num"] = df.groupby([cat])["id"].transfrom("count")
            return df
        else:
            return df.groupby([category_cols])["id"].count().reset_index(name="count")


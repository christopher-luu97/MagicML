from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC):
    
    @abstractmethod
    def load_data(self, input_dir:str):
        """
        General class to load_data

        Args:
            input_dir (str): File path to the file to be used
        """
        pass

    @abstractmethod
    def process_data(self):
        """
        Method execute all processing of data
        There should be internal methods that are called here.
        """
        pass

class DataLoaderPD(DataLoader):
    """
    Dataloader for tabular data to be used through Pandas (PD)

    Args:
        DataLoader (Object): _description_
    """
    def __init__(self, dev:bool = False):
        self.dev = dev

    def load_data(self, input_dir:str):
        """
        Load csv data into a pandas dataframe
        Args:
            input_dir (str): path to input file
        """
        if input_dir[-3:] != "csv":
            print("Input file is not csv!")
            return
        else:
            return pd.read_csv(input_dir)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        """
        pass

    def check_nan(self, df: pd.DataFrame, col_list:list, fill_na:bool = True) -> pd.DataFrame:
        """
        Check for nan and handle it across all columns and rows

        Args:
            df (pd.DataFrame): Input dataframe
            col_list (col): Column name to handle
            fill_na (bool): Fill NA of given columns to 0. Default set to True.
        """
        if fill_na:
            for col in col_list:
                df[col] = df[col].fillna(0) 
        return df

    def check_distribution(self):
        """
        Check distribution of data and handle appropriately
        """
        pass

    def check_outliers(self):
        """
        Check for outliers and handle as needed
        """
        pass

    def convert_datetime(self, df: pd.DataFrame, col_list:list) -> pd.DataFrame:
        """
        Convert appropriate columns to datetime as necessary

        Args:
            df (pd.DataFrame): Input dataframe
            col_list (list): Columns that are datetime

        Returns:
            df (pd.DataFrame): Dataframe with converted datetime columns
        """
        for col in col_list:
            df.loc[:, col] = df[col].apply(lambda s:pd.to_datetime(s))
        return df

# To be implemented
class DataLoaderAudio(DataLoader):
    """
    Dataloader class for audio data

    Args:
        DataLoader (_type_): _description_
    """
    def load_data(self, input_dir:str):
        """_summary_

        Args:
            input_dir (str): path to input file
        """ 
        pass
    
    def process_data(self):
        pass
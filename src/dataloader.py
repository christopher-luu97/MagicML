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

    def check_nan(self):
        """
        Check for nan and handle it across all columns and rows
        """
        pass

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
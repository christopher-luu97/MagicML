import io
import pandas as pd

class TDProcessor():
    """
    This class processes tabular data using pandas dataframes
    """
    def df_info(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get basic info on the input dataframe

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Information of dataframe column
        """
        df.columns = df.columns.str.replace(' ', '_')
        buffer = io.StringIO() 
        df.info(buf=buffer)
        s = buffer.getvalue() 

        df_info = s.split('\n')

        counts = []
        names = []
        nn_count = []
        dtype = []
        for i in range(5, len(df_info)-3):
            line = df_info[i].split()
            counts.append(line[0])
            names.append(line[1])
            nn_count.append(line[2])
            dtype.append(line[4])

        df_info_dataframe = pd.DataFrame(data = {'#':counts, 'Column':names, 'Non-Null Count':nn_count, 'Data Type':dtype})
        return df_info_dataframe.drop('#', axis = 1)

    def df_isnull(df:pd.DataFrame) -> pd.DataFrame:
        """
        Get the percentage of null values per column

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Table showing % of null values per column
        """
        res = pd.DataFrame(df.isnull().sum()).reset_index()
        res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
        res['Percentage'] = res['Percentage'].astype(str) + '%'
        return res.rename(columns = {'index':'Column', 0:'Number of null values'})

    def number_of_outliers(df:pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the number of outliers per column

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe matching column names to the number of outliers they contain
        """
        df = df.select_dtypes(exclude = ['object','bool'])
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        
        ans = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
        df = pd.DataFrame(ans).reset_index().rename(columns = {'index':'column', 0:'count_of_outliers'})
        return df 

    def color_coding(row, condition):
        """_summary_

        Args:
            row (_type_): _description_
            condition (_type_): _description_

        Returns:
            _type_: _description_
        """
        return ['background-color:red'] * len(
            row) if row.col1 == condition else ['background-color:green'] * len(row)
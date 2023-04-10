import pandas as pd
import pytest
from src.tabularDataProcessor import TDProcessor

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [0.1, 0.2, 0.3, 0.4, 0.5],
        'C': ['foo', 'bar', 'baz', 'qux', 'quux'],
        'D': [True, False, True, False, True],
        'E': [1, None, 3, None, 5]
    })

def test_df_info(sample_df):
    result = TDProcessor.df_info(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 3)
    assert result.columns.tolist() == ['Column', 'Non-Null Count', 'Data Type']
    assert result['Column'].tolist() == ['A', 'B', 'C', 'D', 'E']
    assert result['Non-Null Count'].tolist() == [5, 5, 5, 5, 3]
    assert result['Data Type'].tolist() == ['int64', 'float64', 'object', 'bool', 'int64']

def test_df_isnull(sample_df):
    result = TDProcessor.df_isnull(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 3)
    assert result.columns.tolist() == ['Column', 'Number of null values', 'Percentage']
    assert result['Column'].tolist() == ['A', 'B', 'C', 'D', 'E']
    assert result['Number of null values'].tolist() == [0, 0, 0, 0, 2]
    assert result['Percentage'].tolist() == ['0.0%', '0.0%', '0.0%', '0.0%', '40.0%']

def test_number_of_outliers(sample_df):
    result = TDProcessor.number_of_outliers(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert result.columns.tolist() == ['column', 'count_of_outliers']
    assert result['column'].tolist() == ['A', 'B']
    assert result['count_of_outliers'].tolist() == [0, 0]
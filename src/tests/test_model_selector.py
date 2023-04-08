import os
import pandas as pd
from .dataloader import DataLoaderPD
from .calculator import CalculatorPD
from .model_selector import RegressionModelSelector

def test_feature_creation():
    holidays = os.path.join(os.getcwd(), "data","input","holidays.csv")
    airfares = os.path.join(os.getcwd(), "data","input","airfares.csv")
    
    # Load data
    dl = DataLoaderPD()
    holidays_df = dl.load_data(holidays)
    airfares_df = dl.load_data(airfares)
    
    # Perform logic
    calc = CalculatorPD(airfares_df, holidays_df)
    df = calc.join_airfare_holidays()

    rms = RegressionModelSelector(df)
    rms.feature_creation()
    
    expected_result = pd.read_csv(os.path.join(os.getcwd(), "data","test","rms_df_test.csv"))
    pd.testing.assert_frame_equal(rms.df, expected_result)
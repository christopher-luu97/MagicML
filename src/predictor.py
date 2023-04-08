import os
import pickle

class Predictor():
    """
    Class to run predictions from a model
    """
    def __init__(self):
        self.get_model()
        
    def get_model(self, model_name:str):
        """
        Retrieve the model from a specified location

        Args:
            model_name (str): Name of model that was saved
        """
        basedir = os.getcwd()
        model_path = os.path.join(basedir,"models",f"{model_name}.pkl") # Assumes xgb_reg.pkl to be used
        self.model = pickle.load(open(model_path, "rb"))
        
    
    def run(self, df):
        """
        Executor with results
        """
        results = self.model.predict(df)
        df['results'] = results

        return df
    
    def save(self, df, output_name:str):
        """
        Save results as csv

        Args:
            df (pd.DataFrame): Input dataframe
            output_name (str): Output name to save file to
        """
        df.to_csv(f"{output_name}.csv")
        print("Saved to csv!")
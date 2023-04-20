import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, RandomForestRegressor, 
                                BaggingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pickle
import os

import streamlit as st
import pandas as pd
import plotly.express as px

import src.util as functions
from src.pageBuilder import PageBuilderInterface
from src.tabularDataProcessor import TDProcessor


class ModelBuilderPage(PageBuilderInterface):
    """
    Contains the logic to present everything requried for the EDA page

    Args:
        PageBuilderInterface (ABC): Interface class
    """
    st.set_page_config(layout = "wide", page_title = 'Model Builder')

    def __init__(self):
        self.TDP = TDProcessor() # Processor class
        
        # Set session states for persistence
        # Done to allow for OOP instead of procedural programming
        if 'dataset' not in st.session_state:
            st.session_state['dataset'] = pd.DataFrame()
        
        if 'visuals' not in st.session_state:
            st.session_state['visuals'] = []

        if 'num_columns' not in st.session_state:
            st.session_state['num_columns'] = []
        
        if 'cat_columns' not in st.session_state:
            st.session_state['cat_columns'] = []
    
    def app(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        self.set_config()

    def set_config(self):
        """
        Set the page layout
        """
        st.header("Exploratory Data Analysis Tool")
        functions.space()

        st.sidebar.header('Import Dataset to Use Available Features: ')
        st.write('<p style="font-size:130%">Import Dataset (CSV only)</p>', unsafe_allow_html=True)

        use_defo = st.checkbox('Use example Dataset', value=True)

        if use_defo:
            diabetes = load_diabetes()
            X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            Y = pd.Series(diabetes.target, name='response')
            dataset = pd.concat( [X,Y], axis=1 )
        
        if "dataset" not in st.session_state:
            st.session_state["dataset"] = dataset
            df = st.session_state['dataset']
        else:
            df = dataset

        st.markdown('The **Diabetes** dataset is used as the example.')
        st.write(df.head(5))

    def dataset_loader(self):
        """
        Load a datset from the user's file upload
        """
        dataset = st.file_uploader(
                        label="Choose your input file",
                        type=['csv'],
                        accept_multiple_files=False,
                        )  
        if dataset is not None: # Occurs once user uploads
            self.dataset = pd.read_csv(dataset)
            st.session_state["dataset"] = self.dataset
            self.select_data(self.dataset)
            self.setup_data()
    
    def use_default_data(self):
        """
        Load the default diabetes dataset that comes with sklearn
        """
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        dataset = pd.concat( [X,Y], axis=1 )
        
        st.session_state["dataset"] = dataset
        self.select_data(dataset)
        self.setup_data()

     
    def check_data_option(self):
        """
        Check if user wants to use default dataset or load their own
        """
        use_defo = st.checkbox('Use example Dataset')
        if use_defo:
            self.use_default_data()
        else:
            self.dataset_loader()
    
    def select_X_data(self, dataset):
        all_cols = st.session_state["dataset"].columns.tolist()
        st.markdown('')
        st.markdown('**_Features_** you want to use')
        features_selected = st.multiselect("", all_cols)
        st.session_state['X_dataset'] = st.session_state['dataset'][features_selected]

    def select_Y_data(self, dataset):
        all_cols = st.session_state["dataset"].columns.tolist()
        st.markdown('')
        st.markdown('**_Target_** you want to predict')
        target_selected = st.multiselect("", all_cols)
        st.session_state['Y_dataset'] = st.session_state['dataset'][target_selected]


# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Hyperparameter Optimization App',
    layout='wide')

#---------------------------------#
st.write("""
# The Machine Learning Hyperparameter Optimization App
**(Regression Edition)**
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
st.sidebar.header('Set Parameters')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10,50), 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
st.sidebar.write('---')
parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1,3), 1)
st.sidebar.number_input('Step size for max_features', 1)
st.sidebar.write('---')
parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

st.sidebar.subheader('General Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error'])
parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('Dataset')

#---------------------------------#
# Model building

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href

def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    #X_train.shape, Y_train.shape
    #X_test.shape, Y_test.shape

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)

    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)

    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( metrics.r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( metrics.mean_squared_error(Y_test, Y_pred_test) )

    st.write("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    #-----Plot-----#
    layout = go.Layout(
            xaxis=go.layout.XAxis(
              title=go.layout.xaxis.Title(
              text='n_estimators')
             ),
             yaxis=go.layout.YAxis(
              title=go.layout.yaxis.Title(
              text='max_features')
            ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='Hyperparameter tuning',
                      scene = dict(
                        xaxis_title='n_estimators',
                        yaxis_title='max_features',
                        zaxis_title='R2'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    #-----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x,y,z], axis=1)
    st.markdown(filedownload(grid_results), unsafe_allow_html=True)

#---------------------------------#
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    use_defo = st.checkbox('Use example Dataset', value=True)

    if use_defo:
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        dataset = pd.concat( [X,Y], axis=1 )

    if "dataset" not in st.session_state:
        st.session_state["dataset"] = dataset
        df = st.session_state['dataset']
    else:
        df = dataset

    st.markdown('The **Diabetes** dataset is used as the example.')
    st.write(df.head(5))

    all_cols = st.session_state["dataset"].columns.tolist()
    st.markdown('')
    st.markdown('**_Features_** you want to use')
    features_selected = st.multiselect("", all_cols)
    st.session_state['X_dataset'] = st.session_state['dataset'][features_selected]

    st.markdown('')
    st.markdown('**_Target_** you want to predict')
    target_selected = st.multiselect("", all_cols)
    st.session_state['Y_dataset'] = st.session_state['dataset'][target_selected]

    st.write(st.session_state['Y_dataset'])
    st.write(st.session_state['X_dataset'])

    if st.checkbox('Build model', value=False):
        build_model(df)
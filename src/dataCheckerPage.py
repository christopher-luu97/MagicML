import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_diabetes

import src.util as functions
from src.pageBuilder import PageBuilderInterface
from src.tabularDataProcessor import TDProcessor


class DataCheckerPage(PageBuilderInterface):
    """
    Contains the logic to present everything requried for the EDA page

    Args:
        PageBuilderInterface (ABC): Interface class
    """
    st.set_page_config(layout = "wide", page_title = 'EDA')

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
        """
        Overrides PageBuilderInterface abstract method
        Executes everything necessary for the page
        """
        self.set_config()
        self.check_data_option()
        self.get_info()
        self.get_NA_info()
        self.get_descriptive_analysis()
        self.get_target_analysis()
        self.get_dist_numeric_cols()
        self.get_count_cat_cols()
        self.get_box_plot()
        self.get_outlier()
        self.variance_target_cat_cols()

    def set_config(self):
        """
        Set the page layout
        """
        st.header("Exploratory Data Analysis Tool")
        functions.space()

        st.sidebar.header('Import Dataset to Use Available Features: ')
        st.write('<p style="font-size:130%">Import Dataset (CSV only)</p>', unsafe_allow_html=True)
        st.session_state['dataset'] = pd.DataFrame()

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
    
    def select_data(self, dataset):
        option1=st.sidebar.radio(
            'What variables do you want to include in the report?',
            ('All variables', 'A subset of variables'))
            
        if option1=='All Variables':
            st.session_state["dataset"] = dataset
        
        elif option1=='A subset of variables':
            var_list=list(st.session_state['dataset'].columns)
            option3=st.sidebar.multiselect(
                'Select variable(s) you want to include in the report.',
                var_list)
            st.session_state["dataset"] = dataset[option3]


     
    def check_data_option(self):
        """
        Check if user wants to use default dataset or load their own
        """
        use_defo = st.checkbox('Use example Dataset')
        if use_defo:
            self.use_default_data()
        else:
            self.dataset_loader()

     
    def setup_data(self):
        """
        This function sets up the data for the analysis by getting the dataset from the session state and displaying it 
        along with the dimensions of the dataset. It also allows the user to select which visualizations they want to see, 
        stores them in the session state and filters the numerical and categorical columns, storing them in the session state as well.
        """
        df = st.session_state['dataset']
        if df.empty:
            pass
        else:
            st.subheader('Dataframe:')
            n, m = df.shape
            st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
            st.dataframe(df)

        all_visuals = ['Info', 
                       'NA Info', 
                       'Descriptive Analysis', 
                       'Target Analysis', 
                    'Distribution of Numerical Columns', 
                    'Count Plots of Categorical Columns', 
                    'Box Plots', 'Outlier Analysis', 
                    'Variance of Target with Categorical Columns']
        functions.sidebar_space(3)  

        visuals = st.sidebar.multiselect("Choose which visualizations you want to see", all_visuals)
        
        st.session_state['visuals'] = visuals 

        exclude_cols = ['object', 'bool']
        include_cols = ['object']
        num_columns = df.select_dtypes(exclude = exclude_cols).columns
        cat_columns = df.select_dtypes(include = include_cols).columns

        st.session_state['num_columns'] = num_columns
        st.session_state['cat_columns'] = cat_columns

     
    def get_info(self):       
        """
        This function displays the data types and the number of non-null values per column of the dataset.
        """
        if 'Info' in st.session_state['visuals']:
            st.subheader('Info:')
            st.write("""
                    Here we observe the datatypes and the number of non-null values per column
                    """)
            c1, c2, c3 = st.columns([1, 2, 1])
            df = st.session_state['dataset']
            c2.dataframe(TDProcessor.df_info(df))

     
    def get_NA_info(self):
        """
        This function displays the number of missing values for each column in the dataset, if any.
        """
        if 'NA Info' in st.session_state['visuals']:
            st.subheader('NA Value Information:')
            df = st.session_state['dataset']
            if df.isnull().sum().sum() == 0:
                st.write('There are not any NA value in your dataset.')
            else:
                c1, c2, c3 = st.columns([1, 2, 1])
                c2.dataframe(TDProcessor.df_isnull(df))
                functions.space(2)
                
     
    def get_descriptive_analysis(self):
        """
        This function displays the descriptive statistics of the numerical columns in the dataset.
        """
        if 'Descriptive Analysis' in st.session_state['visuals']:
            st.subheader('Descriptive Analysis:')
            df = st.session_state['dataset']
            st.dataframe(df.describe(include = 'all'))
    
     
    def get_target_analysis(self):
        """
        This function displays the histogram of the target column in the dataset.
        """
        if 'Target Analysis' in st.session_state['visuals']:
            df = st.session_state['dataset']
            st.subheader("Select target column:")    
            target_column = st.selectbox("", df.columns, index = len(df.columns) - 1)
        
            st.subheader("Histogram of target column")
            fig = px.histogram(df, x = target_column)
            c1, c2, c3 = st.columns([1, 2, 1])
            c1.plotly_chart(fig)
    
    def __plotter(self, df, selected_cols, name):
        """
        Plots the given columns in a given plot type for a given DataFrame.

        Args:
        df (pandas.DataFrame): The DataFrame to be plotted.
        selected_cols (list): The list of columns to be plotted.
        name (str): The plot type to be used ('hist', 'count_cat', or 'box_plot').
        """
        i = 0
        while (i < len(selected_cols)):
            c1, c2 = st.columns(2)
            for j in [c1, c2]:

                if (i >= len(selected_cols)):
                    break

                if name == 'hist':
                    fig = px.histogram(df, x = selected_cols[i])
                elif name == 'count_cat':
                    fig = px.histogram(df, x = selected_cols[i], color_discrete_sequence=['indianred'])
                elif name == 'box_plot':
                    fig = px.box(df, y = selected_cols[i])
                
                j.plotly_chart(fig, use_container_width = True)
                i += 1

    def __missing_cols(self, cols):
        """
        Checks if the DataFrame has any columns of the specified type and prints a message if it does not.

        Args:
        cols (str): The type of columns to check for ('num' for numerical or 'cat' for categorical).
        """
        if cols=='num':
            if len(st.session_state['num_columns']) == 0 and not(st.session_state['dataset'].empty):
                st.write('There area no numerical columns in the data.')
        elif cols=='cat':
            if len(st.session_state['cat_columns']) == 0 and not(st.session_state['dataset'].empty):
                st.write('There area no categorical columns in the data.')

    def get_dist_numeric_cols(self):
        """
        Displays the distribution of selected numerical columns in the DataFrame.
        """
        if 'Distribution of Numerical Columns' in st.session_state['visuals']:
            df = st.session_state['dataset']
            selected_cols = functions.sidebar_multiselect_container(
                            'Choose columns for Distribution plots:',
                            st.session_state['num_columns'],
                            'Distribution')
            st.subheader('Distribution of numerical columns')
            self.__plotter(df, selected_cols, 'hist')

        self.__missing_cols('num')
                
    def get_count_cat_cols(self):
        """
        Displays the count plots of selected categorical columns in the DataFrame.
        """

        if 'Count Plots of Categorical Columns' in st.session_state['visuals']:
            if (len(st.session_state['cat_columns']) != 0):
                df = st.session_state['dataset']
                selected_cols = functions.sidebar_multiselect_container(
                    'Choose columns for Count plots:', 
                    st.session_state['cat_columns'], 
                    'Count')
                st.subheader('Count plots of categorical columns')
                self.__plotter(df, selected_cols, 'count_cat')
            else:
                self.__missing_cols('cat')

    def get_box_plot(self):
        """
        Displays the box plots of selected numerical columns in the DataFrame.
        """
        if 'Box Plots' in st.session_state['visuals']:
            df = st.session_state['dataset']   
            selected_cols = functions.sidebar_multiselect_container(
                'Choose columns for Box plots:',
                st.session_state['num_columns'], 
                'Box')
            st.subheader('Box plots')
            self.__plotter(df, selected_cols, 'box_plot')   

        self.__missing_cols('num')
        
    def get_outlier(self):
        """
        Displays the number of outliers in the DataFrame.
        """
        if 'Outlier Analysis' in st.session_state['visuals']:
            df = st.session_state['dataset']   
            st.subheader('Outlier Analysis')
            c1, c2, c3 = st.columns([1, 2, 1])
            c2.dataframe(TDProcessor.number_of_outliers(df))

    def get_cardinality_columns(df, cat_columns):
        """
        Given a pandas DataFrame and a list of categorical columns, separates the columns into high-cardinality and normal-
        cardinality columns based on the number of unique values they have.

        Args:
            df (pandas.DataFrame): A pandas DataFrame containing the categorical columns.
            cat_columns (list): A list of categorical columns.

        Returns:
            tuple: A tuple containing two lists, high_cardi_columns and normal_cardi_columns, which are the high-cardinality
            and normal-cardinality categorical columns, respectively.
        """
        high_cardi_columns = []
        normal_cardi_columns = []
        for i in cat_columns:
            if (df[i].nunique() > df.shape[0] / 10):
                high_cardi_columns.append(i)
            else:
                normal_cardi_columns.append(i)
        return high_cardi_columns, normal_cardi_columns

    def plot_box_plot(df, target_column, color_column):
        """
        Given a pandas DataFrame, a target column, and a color column, creates and displays a box plot using plotly and
        streamlit.

        Args:
            df (pandas.DataFrame): A pandas DataFrame containing the data to plot.
            target_column (str): The name of the column that should be plotted on the y-axis.
            color_column (str): The name of the column that should be used to color the boxes in the plot.
        """
        fig = px.box(df, y=target_column, color=color_column)
        st.plotly_chart(fig, use_container_width=True)

    def plot_histogram(df, target_column, color_column):
        """
        Given a pandas DataFrame, a target column, and a color column, creates and displays a histogram using plotly and
        streamlit.

        Args:
            df (pandas.DataFrame): A pandas DataFrame containing the data to plot.
            target_column (str): The name of the column that should be plotted on the x-axis.
            color_column (str): The name of the column that should be used to color the bars in the plot.
        """
        fig = px.histogram(df, color=color_column, x=target_column)
        st.plotly_chart(fig, use_container_width=True)

    def display_high_cardi_columns(high_cardi_columns):
        """
        Displays the names of the high-cardinality columns that were not plotted due to their high cardinality and prompts
        the user to decide whether or not to plot them anyway.

        Args:
            high_cardi_columns (list): A list of the names of the high-cardinality columns that were not plotted.

        Returns:
            bool: True if the user chooses to plot the high-cardinality columns anyway, False otherwise.
        """
        if len(high_cardi_columns) == 1:
            st.subheader('The following column has high cardinality, that is why its boxplot was not plotted:')
        else:
            st.subheader('The following columns have high cardinality, that is why their boxplots were not plotted:')
        for i in high_cardi_columns:
            st.write(i)
        st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)
        answer = st.selectbox("", ('No', 'Yes'))
        return answer == 'Yes'

    def variance_target_cat_cols(self):
        """
        Displays the variance of the target variable with respect to selected categorical columns in the DataFrame.
        """
        if 'Variance of Target with Categorical Columns' in st.session_state['visuals']:
            df = st.session_state['dataset']
            df_1 = df.dropna()
            cat_columns = st.session_state['cat_columns']
            high_cardi_columns, normal_cardi_columns = self.        get_cardinality_columns(df, cat_columns)

            if len(normal_cardi_columns) == 0:
                st.write('There is no categorical columns with normal cardinality in the data.')
            else:
                st.subheader('Variance of target variable with categorical columns')
                model_type = st.radio('Select Problem Type:',
                                    ('Regression', 'Classification'), 
                                    key='model_type')
                selected_cat_cols = functions.sidebar_multiselect_container('Choose columns for Category Colored plots:',normal_cardi_columns, 'Category')

                if 'Target Analysis' not in st.session_state['visuals']:
                    target_column = st.selectbox("Select target column:",
                                                df.columns,
                                                index=len(df.columns) - 1)

                for column in selected_cat_cols:
                    if model_type == 'Regression':
                        self.plot_box_plot(df_1, target_column, column)
                    else:
                        self.plot_histogram(df_1, target_column, column)

                if high_cardi_columns:
                    if self.display_high_cardi_columns(high_cardi_columns):
                        for i in high_cardi_columns:
                            self.plot_box_plot(df_1, target_column, i)

    # def test(self):
    #     st.session_state
    #     st.write(f"{st.session_state['visuals']}")
    #     st.write(f"{st.session_state['num_columns']}")
    #     st.write(f"{st.session_state['cat_columns']}")

    
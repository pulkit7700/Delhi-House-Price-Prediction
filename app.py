# ---- ----- Importing the libraries ---- 

import streamlit as st
import numpy as np 
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import joblib
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model

# -- Making the Preprocessors

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr_names):
        self.attr_names = attr_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attr_names]

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

from sklearn.pipeline import Pipeline

transformers = Pipeline([
    ('Selection', DataFrameSelector(['Area', 'BHK', 'Bathroom', 'Furnishing', 'Locality', 'Parking',
       'Status', 'Transaction', 'Type', 'Per_Sqft', 'Price'])),
    ('inpute', MostFrequentImputer()),
])

# ---------------------- New Pipeline ---------------------------

# ---- Importing the Dataset -----------------

df = pd.read_csv("MagicBricks.csv")
df = df[['Area', 'BHK', 'Bathroom', 'Furnishing', 'Locality', 'Parking', 'Status', 'Transaction', 'Type', 'Per_Sqft', 'Price']]

# ----- Cleaning the Data ------ 


Cleaned_Data = transformers.fit_transform(df) 


# ----------------- Dividing into dependent and Independent Variables --------------------


model = joblib.load('model.joblib')

st.title("House Price Prediction Delhi :house:")

selected = option_menu(
    menu_title=None,
    options=['Predictior', 'Train your Own Model [Classification]'],
    orientation='horizontal'
)

if selected == 'Predictior':
    Area = st.slider('Area of the house', min_value=0, max_value=2000)
    BHK = st.slider('Enter the house BHK', min_value=1, max_value=10)
    Bathroom = st.slider('Enter Number of bathroom', 1, 7)
    Furnishing = st.selectbox('Enter the Furnishing of the house', options=list(Cleaned_Data["Furnishing"].unique()))
    Locality = st.selectbox('Enter the Locality', list(Cleaned_Data["Locality"].unique()))
    Parking = st.slider("Enter Parking space", 0, 10)
    Status = st.selectbox("Enter Status of the Housing", list(Cleaned_Data["Status"].unique()))
    Transaction = st.selectbox('Enter the Transaction', Cleaned_Data["Transaction"].unique())
    Type = st.selectbox('Enter the Type of House', Cleaned_Data["Type"].unique())
    Per_Sqft = st.slider('Enter the House SquareFeet', 0, 15459)

    columns=['Area', 'BHK', 'Bathroom', 'Furnishing', 'Locality', 'Parking', 'Status', 'Transaction', 'Type', 'Per_Sqft']

    def predict(): 
        row = np.array([Area,BHK,Bathroom,Furnishing,Locality,Parking,Status,Transaction,Type,Per_Sqft]) 
        X = pd.DataFrame([row], columns = columns)
        prediction = model.predict(X)[0]
        st.success("The Price of the Property is {}".format(prediction))
        return prediction

    if st.button('Price Prediction', on_click=predict):
        predict()

if selected == 'Train your Own Model [Classification]':
    with st.sidebar:
        st.image('https://media.giphy.com/media/KeQf3wNoXaft4IhWcW/giphy.gif', width=230)
        st.title('AutoTrainML')
        choice = st.radio("Navigation", ["Upload", 'Profiling', 'Training', 'Download'])
        st.info("This is Automated ML training model designed to automatically train models on any device")

    if os.path.exists('Source_data.csv'):
        df2 = pd.read_csv("Source_data.csv", index_col=None)

    if choice == 'Upload':
        st.title('Upload Your File for Modeling !!')
        file = st.file_uploader('Please Upload your File', accept_multiple_files=False)
        if file:
            df2 = pd.read_csv(file, index_col=None)
            df2.to_csv("Source_data.csv", index=None)
            st.dataframe(df2)


    if choice == 'Profiling':
        st.title("EDA")
        profile_report = df2.profile_report()
        st_profile_report(profile_report)

    if choice == 'Training':
        st.title('Machine Learning AutoTraining')
        target = st.selectbox("Select Your Target", df2.columns)
        if st.button("Train Model"):
            setup(df2, target=target)
            setup_df = pull()
            st.info("ML Settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'best_model')


    if choice == 'Download':
        with open('best_model.pkl', 'rb') as f:
            st.download_button("Download the Model", f, "Trained_model.pkl")


   




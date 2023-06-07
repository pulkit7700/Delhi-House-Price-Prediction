# ---- ----- Importing the libraries ---- 

import streamlit as st
import numpy as np 
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import joblib

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
    options=['Predictior', 'Explore'],
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

    st.button('Price Prediction', on_click=predict)


   




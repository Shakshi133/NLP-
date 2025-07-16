import pandas as pd
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st


@st.cache_resource
def load_model():
    df=pd.read_csv("Youtube_comments.csv")
    model=Pipeline([
    ('tfidf',TfidfVectorizer()),
    ('clf',LogisticRegression())
    ])
    model.fit(df['comment'],df['Label'])
    return model

model=load_model()

st.title("HI THERE, I AM A YOUTUBE COMMENT CLASSIFIER")
st.write("Enter your comment to check if it is toxic or not.")
user_input=st.text_area("Comment:")

if user_input:
    prediction=model.predict([user_input])[0]
    if prediction=='toxic':
        st.error("This comment is toxic")
    else:
        st.success("This comment is supportive")





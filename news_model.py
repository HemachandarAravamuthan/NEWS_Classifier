# Importing the required packages
import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(stop_words = 'english',lowercase=False)

# Function to load the model
def load_model():
    with open("C:\\Users\\hp\\Downloads\\news_classifier_pkl",'rb') as file:
        model=pickle.load(file)
    return model


# streamlit page set up
st.set_page_config(page_title='NEWS Classifier',
                   page_icon=':bar_chart:',
                   layout='wide')
st.header('NEWS Headline Classifier')

# Input
headline=[st.text_input('ENTER THE HEADLINE')]

# Prediction button
if st.button('Predict the category'):
    vec= vector.transform(headline).toarray()
    model_class=load_model()
    st.success(str(list(model_class.predict(vec))[0]).replace('0', 'WELLNESS').replace('1', 'POLITICS')
               .replace('2', 'ENTERTAINMENT').replace('3','TRAVEL').replace('4','STYLE & BEAUTY').replace('5','PARENTING')
               .replace('6','FOOD & DRINK').replace('7','WORLD NNEWS').replace('8','BUSINESS').replace('9','SPORTS'))
    

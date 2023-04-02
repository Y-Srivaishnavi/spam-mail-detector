import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

st.title("Spam Mail Detector")
st.header("Not sure if you've really won that lucky draw? Don't worry, we've got you covered!")

with open('spam_mail_detector.pkl', 'rb') as pickle_model:
    spam_detector = pickle.load(pickle_model)

with open('feature_extraction.pkl', 'rb') as pickle_features:
    required_features = pickle.load(pickle_features)

input_mail = st.text_area("Put in your sus mail here: ")

if input_mail != "":
    mail_as_iterable = [input_mail]
    mail_as_array = required_features.transform(mail_as_iterable)

    st.cache_data.clear()
    prediction_str = ""

    try: 
        prediction = spam_detector.predict(mail_as_array)
        if prediction == [0]:
            prediction_str = 'Spam'
        elif prediction == [1]:
            prediction_str = 'Not Spam'
        else:
            prediction_str = ''
        st.text(prediction_str)

    except Exception as e:
        print(e)
        st.error("Not sure, what went wrong. We'll get back to you shortly!")

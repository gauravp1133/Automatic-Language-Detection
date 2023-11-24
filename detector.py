import pickle
import streamlit as st
st.title("Language Detector Application")

sentence = st.text_input("Please Enter the sentence")

if sentence:    
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    inputdata = vectorizer.transform([sentence])
    confidence_values = model.predict_proba(inputdata)
    confidence = confidence_values.max()

    prediction = model.predict(inputdata)
    st.write("The predicted Language is:", prediction[0])
    st.write("Confidence:", confidence)

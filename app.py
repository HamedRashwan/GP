import streamlit as st
import joblib
import os
from sklearn.metrics import accuracy_score
model_path = 'text_classification_model.joblib'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    # Load the model
    pipeline = joblib.load(model_path)

    # Streamlit interface
    st.title("Text Classification Model")

    st.write("Enter a question and the model will predict the answer.")

    # User input
    question = st.text_input("Question")

    if st.button("Predict"):
        if question:
            prediction = pipeline.predict([question])
            st.write(f"Predicted Answer: {prediction[0]}")
        else:
            st.write("Please enter a question.")

    # Display model accuracy
    accuracy =accuracy_score(prediction) # Assuming you have this accuracy from the training phase
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

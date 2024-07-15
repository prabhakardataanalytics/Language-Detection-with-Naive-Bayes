import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
print(data.head())

# Display the dataset
st.title("Language Detection with Naive Bayes")
st.write("## Dataset")
st.write(data.head())

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Text'])
y = data['language']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f"## Model Accuracy: {accuracy:.2f}")

# Add a text input box for user to enter new text
st.write("## Predict Language for New Text")
user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)
    st.write(f"Predicted Language: {prediction[0]}")
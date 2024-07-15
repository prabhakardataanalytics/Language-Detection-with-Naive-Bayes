# Language-Detection-with-Naive-Bayes
This project demonstrates language detection using a Naive Bayes classifier implemented in Python with Streamlit. It utilizes a dataset containing text samples in different languages to train and evaluate the classifier.

Features:
Dataset: The project loads a dataset containing text samples from various languages.
Vectorization: Text data is vectorized using CountVectorizer from sklearn.feature_extraction.text.
Model Training: A Multinomial Naive Bayes classifier is trained on the vectorized data.
Prediction: Users can input text to predict the language using the trained model.
Evaluation: The model's accuracy is calculated using the test set.
Technologies Used:
Python
Streamlit
Pandas
Scikit-learn (sklearn)
How to Use:
Clone the repository.
Install the necessary dependencies (streamlit, pandas, scikit-learn).
Run the Streamlit app using streamlit run app.py.
Explore the dataset, train the model, and predict languages for new text inputs.

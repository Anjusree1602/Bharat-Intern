import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

# Preprocess the dataset
X = data.drop('species', axis=1)  # features
y = data['species']  # target variable

# Map species to numerical values
y = y.map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Define a function to make predictions
def predict(sepal_length, sepal_width, petal_length, petal_width):
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    return prediction[0]

# Create the Streamlit app
st.title('Iris Classification')
st.markdown('Toy model to play to classify iris flowers into setosa, versicolor, virginica')

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input('Sepal length (cm)', min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input('Sepal width (cm)', min_value=0.0, max_value=6.0, value=3.5, step=0.1)
with col2:
    petal_length = st.number_input('Petal length (cm)', min_value=0.0, max_value=8.0, value=1.4, step=0.1)
    petal_width = st.number_input('Petal width (cm)', min_value=0.0, max_value=3.0, value=0.2, step=0.1)

if st.button('Predict type of Iris'):
    result = predict(sepal_length, sepal_width, petal_length, petal_width)
    st.text(result)

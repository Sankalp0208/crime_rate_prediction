import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Title
st.title("💳 Credit Card Fraud Detection (Random Forest)")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("new_dataset.xlsx")
    return df

df = load_data()
st.subheader("Raw Data")
st.write(df.head())

# Preprocessing
df = df.dropna()  # Drop missing values
le = LabelEncoder()

# Encode categorical features if any
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Features and target
target_col = 'Class'  # Replace with actual target column if different
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Display model performance
st.subheader("Model Performance")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Save model (optional)
joblib.dump(model, "model.pkl")

# Prediction UI
st.subheader("🔍 Predict from Custom Input")

input_data = []
for col in X.columns:
    val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"Predicted Class: {prediction[0]}")

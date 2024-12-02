import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# Sidebar Navigation
st.sidebar.title("Navigation")
# Add a unique key for this radio button to prevent duplication
section = st.sidebar.radio("Choose a section", ["Introduction", "Visualization"], key="main_navigation")

# Define paths to CSV files
csv_files = {
    "Main Dataset": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data1.csv",
    "Dataset 2": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data2.csv",
    "Dataset 3": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data3.csv",
    "Dataset 4": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data4.csv",
    "Dataset 5": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data5.csv",
}

# Load all datasets
dataframes = {}
for name, path in csv_files.items():
    try:
        dataframes[name] = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load {name}: {e}")

# Merge all datasets into one (assuming a common key exists)
try:
    # Define the common key for merging (replace with the correct key, e.g., 'vehicle_id')
    common_key = "vehicle_id"

    # Merge datasets iteratively
    merged_df = dataframes["Main Dataset"]
    for name, df in dataframes.items():
        if name != "Main Dataset":
            merged_df = merged_df.merge(df, on

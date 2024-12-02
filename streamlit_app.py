import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# Define paths to multiple datasets
csv_files = {
    "Main Dataset": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data1.csv",
    "Dataset 2": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data2.csv",
    "Dataset 3": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data3.csv",
    "Dataset 4": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data4.csv",
    "Dataset 5": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data5.csv",
}

# Load datasets
dataframes = {}
for name, path in csv_files.items():
    try:
        dataframes[name] = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load {name}: {e}")

# Display dataset column names for debugging
st.write("### Columns in Each Dataset")
for name, df in dataframes.items():
    st.write(f"**{name}**")
    st.write(df.columns.tolist())

# Merge datasets based on a common key
common_key = "vehicle_id"  # Replace with the appropriate key if needed
try:
    merged_df = dataframes["Main Dataset"]
    for name, df in dataframes.items():
        if name != "Main Dataset":
            if common_key in df.columns:
                merged_df = merged_df.merge(df, on=common_key, how="inner")
            else:
                st.warning(f"Skipping merge with {name} as it lacks the common key '{common_key}'")
    st.success("Datasets successfully merged!")
    st.write(f"Merged Dataset Shape: {merged_df.shape}")
    st.dataframe(merged_df.head())

except Exception as e:
    st.error(f"An error occurred during merging: {e}")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "Visualization"])

# Introduction Section
if section == "Introduction":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    This app demonstrates a machine learning workflow for classifying vehicle transmissions 
    (Automatic or Manual) based on various features like model year, make, mileage, price, and more.
    The dataset contains vehicle listings from Edmonton dealerships and additional related datasets.
    """)
    st.write("### Merged Dataset Preview:")
    st.dataframe(merged_df.head(10))

# Visualization Section
elif section == "Visualization":
    st.title("📊 Data Visualizations")
    st.write("Explore key insights and trends in the dataset through the visualizations below:")

    # Visualization 1: Transmission Type Distribution
    image_url_1 = "plt1.png"
    st.image(image_url_1, caption="Transmission Type Distribution", use_column_width=True)

    # Visualization 2: Price vs Mileage Scatter Plot
    image_url_2 = "plt2.png"
    st.image(image_url_2, caption="Price vs Mileage Scatter Plot", use_column_width=True)

    # Visualization 3: Correlation Heatmap
    image_url_3 = "plt3.png"
    st.image(image_url_3, caption="Feature Correlation Heatmap", use_column_width=True)

    # Visualization 4: Model Year Distribution
    image_url_4 = "plt4.png"
    st.image(image_url_4, caption="Model Year Distribution", use_column_width=True)

    # Visualization 5: Price Distribution by Fuel Type
    image_url_5 = "plt5.png"
    st.image(image_url_5, caption="Price Distribution by Fuel Type", use_column_width=True)

    # Visualization 6: Mileage Boxplot by Transmission Type
    image_url_6 = "plt6.png"
    st.image(image_url_6, caption="Mileage Boxplot by Transmission Type", use_column_width=True)

    # Visualization 7: Price vs Model Year Line Plot
    image_url_7 = "plt7.png"
    st.image(image_url_7, caption="Price vs Model Year Trend", use_column_width=True)

    # Visualization 8: Make Popularity Countplot
    image_url_8 = "plt8.png"
    st.image(image_url_8, caption="Make Popularity Countplot", use_column_width=True)

    st.write("These visualizations provide insights into the dataset and model performance.")

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
section = st.sidebar.radio("Go to", ["Introduction", "Dataset Overview", "Visualization"])

# Define paths to CSV files
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

# Merge datasets based on a common key
common_key = "vehicle_id"
try:
    merged_df = dataframes["Main Dataset"]
    for name, df in dataframes.items():
        if name != "Main Dataset" and common_key in df.columns:
            merged_df = merged_df.merge(df, on=common_key, how="inner")
    st.success("Datasets successfully merged!")
except Exception as e:
    st.error(f"An error occurred during merging: {e}")

# Introduction Section
if section == "Introduction":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    This app demonstrates a machine learning workflow for classifying vehicle transmissions 
    (Automatic or Manual) based on various features like model year, make, mileage, price, and more.
    The dataset contains vehicle listings from Edmonton dealerships and additional related datasets.
    """)

# Dataset Overview Section
elif section == "Dataset Overview":
    st.title("📊 Dataset Overview")
    
    # Display the first 5 rows of main columns
    main_columns = ["vehicle_id", "model_year", "make", "price", "mileage", "transmission_type"]
    available_columns = [col for col in main_columns if col in merged_df.columns]
    
    if available_columns:
        st.write("### Merged Dataset (First 5 Rows of Main Columns):")
        st.dataframe(merged_df[available_columns].head(5))
    else:
        st.warning("Main columns not found in the dataset.")
    
    # Display general statistics
    st.write("### General Statistics:")
    st.write(merged_df.describe())

# Introduction Section
if section == "Introduction":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    This app demonstrates a machine learning workflow for classifying vehicle transmissions 
    (Automatic or Manual) based on various features like model year, make, mileage, price, and more.
    The dataset contains vehicle listings from Edmonton dealerships and additional related datasets.
    """)

    st.write("### Datasets Overview")
    for name, df in dataframes.items():
        st.subheader(name)
        st.write(f"**Shape:** {df.shape}")
        st.dataframe(df.head())

# Visualization Section
elif section == "Visualization":
    st.title("📊 Data Visualizations")
    st.write("""
    Explore key insights and trends in the dataset through the visualizations below. These charts help uncover patterns 
    and relationships between variables in the vehicle transmission dataset.
    """)

    # Visualization 1: Transmission Type Distribution
    st.subheader("1️⃣ Transmission Type Distribution")
    st.image("plt1.png", caption="Proportion of Automatic vs Manual Transmissions", use_column_width=True)
    st.write("""
    This chart shows the distribution of transmission types across the dataset, providing insights into the balance 
    between automatic and manual vehicles.
    """)

    # Visualization 2: Price vs Mileage Scatter Plot
    st.subheader("2️⃣ Price vs Mileage Scatter Plot")
    st.image("plt2.png", caption="Price vs Mileage for Different Vehicles", use_column_width=True)
    st.write("""
    This scatter plot illustrates the relationship between vehicle price and mileage, showing how mileage impacts 
    price variation.
    """)

    # Visualization 3: Feature Correlation Heatmap
    st.subheader("3️⃣ Correlation Heatmap")
    st.image("plt3.png", caption="Correlation Among Dataset Features", use_column_width=True)
    st.write("""
    The heatmap highlights the strength of correlations between features, helping identify key predictors for transmission types.
    """)

    # Visualization 4: Model Year Distribution
    st.subheader("4️⃣ Model Year Distribution")
    st.image("plt 4.png", caption="Distribution of Vehicles by Model Year", use_column_width=True)
    st.write("""
    This chart shows the frequency of vehicles manufactured in each model year, revealing trends in dataset age distribution.
    """)

    # Visualization 5: Price Distribution by Fuel Type
    st.subheader("5️⃣ Price Distribution by Fuel Type")
    st.image("plt5.png", caption="Price Variation Across Fuel Types", use_column_width=True)
    st.write("""
    Analyzing vehicle price distribution across different fuel types provides insights into market segmentation by fuel preferences.
    """)

    # Visualization 6: Mileage Boxplot by Transmission Type
    st.subheader("6️⃣ Mileage Boxplot by Transmission Type")
    st.image("plt6.png", caption="Mileage Distribution for Automatic and Manual Transmissions", use_column_width=True)
    st.write("""
    This boxplot compares the mileage distribution between automatic and manual vehicles, highlighting central tendencies and outliers.
    """)

    # Visualization 7: Price vs Model Year Line Plot
    st.subheader("7️⃣ Price vs Model Year Trend")
    st.image("plt7.png", caption="Average Price Trends by Model Year", use_column_width=True)
    st.write("""
    The line plot showcases how vehicle prices vary over model years, helping identify depreciation or appreciation trends.
    """)

    # Visualization 8: Make Popularity Countplot
    st.subheader("8️⃣ Make Popularity Countplot")
    st.image("plt8.png", caption="Frequency of Vehicle Makes in the Dataset", use_column_width=True)
    st.write("""
    This countplot highlights the most popular vehicle makes in the dataset, showing the relative frequency of each brand.
    """)


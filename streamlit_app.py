import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
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

# Concatenate datasets vertically
try:
    # Start with the main dataset
    merged_df = dataframes["Main Dataset"]
    
    # Concatenate other datasets one by one
    for name, df in dataframes.items():
        if name != "Main Dataset":
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    
except Exception as e:
    st.error(f"An error occurred during merging: {e}")

# Sidebar Navigation with custom 'key' for the radio button to prevent duplicate element IDs
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "Dataset Overview", "Visualization", "Model Preprocessing & Training", "Model Validation", "Power BI Dashboard"], key="section_radio")

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
    main_columns = ["vehicle_id", "dealer_type", "stock_type", "model_year", "model", "make", "price", "mileage", "transmission_type", "certified", "transmission_from_vin", "fuel_type_from_vin", "number_price_changes"]
    available_columns = [col for col in main_columns if col in merged_df.columns]
    
    if available_columns:
        st.write("### First 5 Rows of the Dataset:")
        st.dataframe(merged_df[available_columns].head(5))
    else:
        st.warning("Main columns not found in the dataset.")
    
    # Display general statistics
    st.write("### General Statistics:")
    st.write(merged_df.describe())

# Visualization Section
elif section == "Data Visualization":
    st.title("📊 Data Visualizations")
    st.write("""
    These charts help uncover patterns 
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

# Model Preprocessing & Training Section
elif section == "Model Preprocessing & Training":
    st.title("🧑‍🔬 Model Preprocessing & Training")

    try:
        # Label Encoding for the target variable
        le = LabelEncoder()
        merged_df["transmission_from_vin"] = le.fit_transform(merged_df["transmission_from_vin"])

        # Select features (X) and target (y)
        X = merged_df[["model_year", "mileage", "price"]].dropna()  # Features
        y = merged_df["transmission_from_vin"].loc[X.index]  # Target

        # Data Imbalance Handling using SMOTE (Synthetic Minority Over-sampling Technique)
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)  # Resample the data

        # Display the original and resampled class distribution
        st.write("Original class distribution:")
        st.write(y.value_counts())
        st.write("Resampled class distribution:")
        st.write(y_res.value_counts())

        # Train a Random Forest Classifier
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model Validation: Display performance metrics
        st.write("### Accuracy Score:", accuracy_score(y_test, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.write("### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)

        # Display confusion matrix as a table
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        st.write(cm_df)

        # Save the trained model
        joblib.dump(model, 'model.pkl')

        st.write("""
        The model has been trained and saved. You can use this trained model for further predictions.
        """)
        
    except Exception as e:
        st.error(f"An error occurred during preprocessing or training: {e}")  

# Model Validation Section
elif section == "Model Validation":
    st.title("🔍 Model Validation")
    
    try:
        # Load the trained model
        model = joblib.load('model.pkl')
        
        # Calculate confusion matrix
        X_test, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)[1:3]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Display the classification report
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Display confusion matrix
        st.write("### Confusion Matrix:")
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        st.write(cm_df)
        
    except Exception as e:
        st.error(f"Error during model validation: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("""
    Visualize data insights using an integrated Power BI dashboard.
    """)
 

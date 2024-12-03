import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
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
    merged_df = dataframes["Main Dataset"]
    for name, df in dataframes.items():
        if name != "Main Dataset":
            merged_df = pd.concat([merged_df, df], ignore_index=True)
except Exception as e:
    st.error(f"An error occurred during merging: {e}")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Introduction", "Dataset Overview", "Visualization", "Model Preprocessing & Training", "Model Validation", "Power BI Dashboard"],
    key="section_radio"
)

# Introduction Section
if section == "Introduction":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    This app demonstrates a machine learning workflow for classifying vehicle transmissions 
    (Automatic or Manual) based on various features like model year, make, mileage, price, and more.
    """)

# Dataset Overview Section
elif section == "Dataset Overview":
    st.title("📊 Dataset Overview")
    main_columns = [
        "vehicle_id", "dealer_type", "stock_type", "model_year", "model", "make",
        "price", "mileage", "transmission_type", "certified", "transmission_from_vin",
        "fuel_type_from_vin", "number_price_changes"
    ]
    available_columns = [col for col in main_columns if col in merged_df.columns]

    if available_columns:
        st.write("### First 5 Rows of the Dataset:")
        st.dataframe(merged_df[available_columns].head(5))
    else:
        st.warning("Main columns not found in the dataset.")

    st.write("### General Statistics:")
    st.write(merged_df.describe())

# Visualization Section
elif section == "Visualization":
    st.title("📊 Data Visualizations")
    st.write("""
    These charts help uncover patterns 
    and relationships between variables in the vehicle transmission dataset.
    """)

    st.subheader("1️⃣ Transmission Type Distribution")
    st.image("plt1.png", caption="Proportion of Automatic vs Manual Transmissions", use_column_width=True)

    st.subheader("2️⃣ Price vs Mileage Scatter Plot")
    st.image("plt2.png", caption="Price vs Mileage for Different Vehicles", use_column_width=True)

    st.subheader("3️⃣ Correlation Heatmap")
    st.image("plt3.png", caption="Correlation Among Dataset Features", use_column_width=True)

    st.subheader("4️⃣ Model Year Distribution")
    st.image("plt4.png", caption="Distribution of Vehicles by Model Year", use_column_width=True)

    st.subheader("5️⃣ Price Distribution by Fuel Type")
    st.image("plt5.png", caption="Price Variation Across Fuel Types", use_column_width=True)

    st.subheader("6️⃣ Mileage Boxplot by Transmission Type")
    st.image("plt6.png", caption="Mileage Distribution for Automatic and Manual Transmissions", use_column_width=True)

    st.subheader("7️⃣ Price vs Model Year Trend")
    st.image("plt7.png", caption="Average Price Trends by Model Year", use_column_width=True)

    st.subheader("8️⃣ Make Popularity Countplot")
    st.image("plt8.png", caption="Frequency of Vehicle Makes in the Dataset", use_column_width=True)

# Model Preprocessing & Training Section
elif section == "Model Preprocessing & Training":
    st.title("🧑‍🔬 Model Preprocessing & Training")
    try:
        le = LabelEncoder()
        merged_df["transmission_from_vin"] = le.fit_transform(merged_df["transmission_from_vin"])
        X = merged_df[[
            "dealer_type", "stock_type", "mileage", "price", "model_year",
            "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
        ]].dropna()
        y = merged_df["transmission_from_vin"].loc[X.index]
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("### Accuracy Score:", accuracy_score(y_test, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        joblib.dump(model, "vehicle_transmission_model.pkl")
    except Exception as e:
        st.error(f"Error during model preprocessing/training: {e}")

# Model Validation Section
elif section == "Model Validation":
    st.title("🔍 Model Validation")
    try:
        model = joblib.load("vehicle_transmission_model.pkl")
        st.success("Model loaded successfully.")

        le = LabelEncoder()

        # Prepare the data for validation
        X = merged_df[[
            "dealer_type", "stock_type", "mileage", "price", "model_year",
            "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
        ]].dropna()

        y = merged_df["transmission_from_vin"].loc[X.index]

        # Encode categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict and evaluate
        y_pred = model.predict(X_scaled)

        st.write("### Accuracy Score:", accuracy_score(y, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y, y_pred))
        st.write("### Confusion Matrix:")
        st.write(confusion_matrix(y, y_pred))
        
    except FileNotFoundError:
        st.warning("Model file not found. Please train the model first.")
    except Exception as e:
        st.error(f"Error during model validation: {e}")


# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Link to Power BI dashboard or embedded visualization.")

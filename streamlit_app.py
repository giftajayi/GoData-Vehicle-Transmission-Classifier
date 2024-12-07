import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the uploaded dataset
data_file_path = "/mnt/data/Cleaned_data1.csv"
df = load_data(data_file_path)

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "EDA",
        "Feature Engineering and Model Training",
        "Model Prediction",
    ],
)

# Dashboard Section
if section == "Dashboard":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write(
        """
        This application predicts the type of transmission (Automatic or Manual) 
        for vehicles based on various features.
        """
    )

# EDA Section elif section == "EDA": st.title("📊 Exploratory Data Analysis (EDA)") st.subheader("Dataset Information") st.image("info1.jpeg", caption="Dataset Overview - Part 1") st.image("info2.jpeg", caption="Dataset Overview - Part 2") st.subheader("Visualizations") st.image("chart7.jpeg", caption="Transmission Distribution (Auto vs Manual)") st.image("chart2.png", caption="Price vs Mileage Scatter Plot") st.image("plt3.png", caption="Correlation Heatmap")

# Feature Engineering and Model Training Section
elif section == "Feature Engineering and Model Training":
    st.title("🧑‍🔬 Feature Engineering and Model Training")

    # Encode categorical variables
    st.write("### Encoding Categorical Variables")
    le = LabelEncoder()
    df["transmission_from_vin"] = le.fit_transform(df["transmission_from_vin"])
    joblib.dump(le, "models/label_encoder.pkl")  # Save the label encoder

    st.write("### Handling Missing Data")
    df = df.dropna()  # Drop rows with missing values

    # Select features and target
    X = df[
        [
            "dealer_type",
            "stock_type",
            "mileage",
            "price",
            "model_year",
            "make",
            "model",
            "certified",
            "fuel_type_from_vin",
            "number_price_changes",
        ]
    ]
    y = df["transmission_from_vin"]

    # Encode categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")  # Save the scaler

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train the model with class weighting
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "models/vehicle_transmission_model.pkl")

    # Evaluate the model
    y_pred = model.predict(X_test)
    st.write("### Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    st.success("Model training completed and saved successfully.")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    try:
        model = joblib.load("models/vehicle_transmission_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        st.write("Model and files loaded successfully.")
    except Exception as e:
        st.error(f"Error loading files: {e}")

    st.subheader("Enter Vehicle Details:")

    dealer_type = st.selectbox("Dealer Type", df["dealer_type"].unique())
    stock_type = st.selectbox("Stock Type", df["stock_type"].unique())
    mileage = st.number_input("Mileage", min_value=0)
    price = st.number_input("Price", min_value=0)
    model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
    make = st.selectbox("Make", df["make"].unique())
    model = st.selectbox("Model", df["model"].unique())
    certified = st.radio("Certified", ["Yes", "No"])
    fuel_type = st.selectbox("Fuel Type", df["fuel_type_from_vin"].unique())
    price_changes = st.number_input("Number of Price Changes", min_value=0)

    input_data = pd.DataFrame(
        [
            {
                "dealer_type": dealer_type,
                "stock_type": stock_type,
                "mileage": mileage,
                "price": price,
                "model_year": model_year,
                "make": make,
                "model": model,
                "certified": 1 if certified == "Yes" else 0,
                "fuel_type_from_vin": fuel_type,
                "number_price_changes": price_changes,
            }
        ]
    )

    st.write("Input Data for Prediction:")
    st.write(input_data)

    if st.button("Generate Prediction"):
        try:
            for col in input_data.select_dtypes(include=["object"]).columns:
                input_data[col] = label_encoder.transform(input_data[col].astype(str))

            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)
            predicted_transmission = label_encoder.inverse_transform(prediction)
            st.write(f"### Predicted Transmission: {predicted_transmission[0]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

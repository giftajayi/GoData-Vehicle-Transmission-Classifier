import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load Data Function
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    try:
        if uploaded_file:
            # Load from uploaded file
            return pd.read_csv(uploaded_file)
        elif file_path and os.path.exists(file_path):
            # Load from file path
            return pd.read_csv(file_path)
        else:
            st.error("File not found. Please upload the dataset or provide the correct path.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Optimize DataFrame Memory Usage
@st.cache_data
def optimize_dataframe(df):
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

# Ensure the models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

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
        The primary objective of this project is to develop a machine learning model 
        to predict a vehicle's transmission type, enhancing Go Auto’s decision-making 
        and marketing strategies.
        """
    )

# File Upload and Data Loading
elif section == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    data_file_path = "Cleaned_data1.csv"  # Default file path

    # Load the dataset
    df = load_data(data_file_path, uploaded_file)
    if df is not None:
        df = optimize_dataframe(df)
        st.write("Dataset Loaded Successfully!")
        st.write(df.head())
        st.write(f"### Dataset Summary:")
        st.write(df.describe())
    else:
        st.error("No dataset available for analysis.")

# Feature Engineering and Model Training
elif section == "Feature Engineering and Model Training":
    st.title("🧑‍🔬 Feature Engineering and Model Training")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    data_file_path = "Cleaned_data1.csv"  # Default file path

    # Load the dataset
    df = load_data(data_file_path, uploaded_file)
    if df is not None:
        try:
            # Encoding categorical variables
            le = LabelEncoder()
            df["transmission_from_vin"] = le.fit_transform(df["transmission_from_vin"])

            # Save the label encoder
            joblib.dump(le, "models/label_encoder.pkl")

            # Drop missing values
            df = df.dropna()

            # Feature Selection
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

            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Train the model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            st.write("### Initial Model Evaluation:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            st.write("### Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Save the model and scaler
            joblib.dump(model, "models/vehicle_transmission_model.pkl")
            joblib.dump(scaler, "models/scaler.pkl")
            joblib.dump(X.columns, "models/original_columns.pkl")

            st.success("Model trained and saved successfully.")
        except Exception as e:
            st.error(f"Error during feature engineering or model training: {e}")
    else:
        st.error("Dataset not available. Please upload a valid dataset.")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    try:
        # Load saved model and related files
        model = joblib.load("models/vehicle_transmission_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        original_columns = joblib.load("models/original_columns.pkl")
        st.write("Model and files loaded successfully.")
    except Exception as e:
        st.error(f"Error loading files: {e}")

    # Input form for prediction
    st.subheader("Enter Vehicle Details:")
    dealer_type = st.text_input("Dealer Type")
    stock_type = st.text_input("Stock Type")
    mileage = st.number_input("Mileage", min_value=0.0)
    price = st.number_input("Price", min_value=0.0)
    model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
    make = st.text_input("Make")
    model = st.text_input("Model")
    certified = st.radio("Certified", ["Yes", "No"])
    fuel_type = st.text_input("Fuel Type")
    price_changes = st.number_input("Number of Price Changes", min_value=0)

    # Prepare input data
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

    # Display input data
    st.write("Input Data for Prediction:")
    st.write(input_data)

    if st.button("Generate Prediction"):
        try:
            # Ensure correct column order and missing columns
            input_data = input_data.reindex(columns=original_columns, fill_value=0)

            # Encode and scale input data
            for col in input_data.select_dtypes(include=["object"]).columns:
                input_data[col] = label_encoder.transform(input_data[col].astype(str))
            scaled_input = scaler.transform(input_data)

            # Predict and decode the output
            prediction = model.predict(scaled_input)
            predicted_transmission = label_encoder.inverse_transform(prediction)
            st.write(f

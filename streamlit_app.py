import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')  # Create directory if not exists

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Dashboard", "EDA", "Model Training", "Model Prediction", "Power BI Dashboard"],
)

# Dashboard Section
if section == "Dashboard":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write(
        """
        The primary objective of this project is to develop a machine learning model 
        that can reliably predict whether a vehicle has an automatic or manual transmission. 
        This supports Go Auto's decision-making, inventory management, and marketing efforts.
        """
    )

# EDA Section
elif section == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write(
        """
        In the initial phase, we performed data cleaning, addressed missing values, resolved inconsistencies, 
        and analyzed relationships between features like vehicle year, price, and mileage. 
        Insights included patterns such as newer vehicles being more likely to have automatic transmissions.
        """
    )
    st.subheader("Dataset Visualizations")
    st.write("**Key Visualizations** (Add your EDA images or charts here)")

# Model Training Section
elif section == "Model Training":
    st.title("🧑‍🔬 Model Training")
    st.write(
        """
        Models such as Logistic Regression, Random Forest, and XGBoost were trained using pipelines for preprocessing 
        (handling missing values, scaling, encoding) with 5-fold cross-validation. Results were compared for accuracy and stability.
        """
    )
    st.write("Add visualizations or performance metrics here if available.")
    st.warning("No performance metrics visualization available yet.")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")
    try:
        # Load model and related files
        model = joblib.load('models/vehicle_transmission_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')
        le_transmission = joblib.load('models/le_transmission.pkl')
        original_columns = joblib.load('models/original_columns.pkl')
        merged_df = pd.read_csv("models/merged_dataset.csv")  # Add the path to your dataset
    except Exception as e:
        st.error(f"Error loading files: {e}")
        model, scaler, encoders, le_transmission, original_columns, merged_df = None, None, None, None, None, None

    if model:
        st.subheader("Enter Vehicle Details:")
        dealer_type = st.selectbox("Dealer Type", merged_df['dealer_type'].unique())
        stock_type = st.selectbox("Stock Type", merged_df['stock_type'].unique())
        mileage = st.number_input("Mileage", min_value=0)
        price = st.number_input("Price", min_value=0)
        model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
        make = st.selectbox("Make", merged_df['make'].unique())
        available_models = merged_df[merged_df['make'] == make]['model'].unique()
        model_input = st.selectbox("Model", available_models)
        certified = st.radio("Certified", ["Yes", "No"])
        fuel_type = st.selectbox("Fuel Type", merged_df['fuel_type_from_vin'].unique())
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
                    "model": model_input,
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
                input_df = input_data.reindex(columns=original_columns, fill_value=0)
                for col, encoder in encoders.items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = encoder.transform(input_df[col].astype(str))
                        except KeyError:
                            input_df[col] = encoder.transform([input_df[col].mode()[0]])[0]
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)
                transmission_mapping = {0: "Automatic", 1: "Manual"}
                predicted_transmission = transmission_mapping.get(prediction[0], "Unknown")
                st.write(f"### Predicted Transmission: {predicted_transmission}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Click [here](https://app.powerbi.com) to view the dashboard.")

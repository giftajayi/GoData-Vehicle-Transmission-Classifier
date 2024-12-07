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

# URLs for datasets
csv_urls = [
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data1.csv",
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data2.csv",
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data3.csv",
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data4.csv",
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data5.csv",
]

# Cache loading and merging of datasets
@st.cache_data
def load_and_merge_data():
    try:
        dfs = [pd.read_csv(url) for url in csv_urls]
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading datasets: {e}")

@st.cache_data
def optimize_dataframe(df):
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

merged_df = optimize_dataframe(load_and_merge_data())

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')  # Create directory if not exists

# Helper function to encode categorical features using LabelEncoder
def encode_features(df, encoders=None):
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in df.select_dtypes(include=['object']).columns}
    
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    
    return df, encoders

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "EDA",
        "Feature Engineering and Model Training",
        "Model Prediction",
        "Power BI Dashboard",
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

# EDA Section
elif section == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.subheader("Dataset Information")
    st.image("info1.jpeg", caption="Dataset Overview - Part 1")
    st.image("info2.jpeg", caption="Dataset Overview - Part 2")
    st.subheader("Visualizations")
    st.image("chart7.jpeg", caption="Transmission Distribution (Auto vs Manual)")
    st.image("chart2.png", caption="Price vs Mileage Scatter Plot")
    st.image("plt3.png", caption="Correlation Heatmap")

# Feature Engineering and Model Training Section
if section == "Feature Engineering and Model Training":
    st.title("🧑‍🔬 Feature Engineering and Model Training")

    try:
        # 1. Encoding categorical variables using LabelEncoder
        merged_df, encoders = encode_features(merged_df)

        le_transmission = LabelEncoder()
        merged_df["transmission_from_vin"] = le_transmission.fit_transform(merged_df["transmission_from_vin"])

        # Save the encoders and label encoder for later use
        joblib.dump(encoders, "models/encoders.pkl")
        joblib.dump(le_transmission, "models/le_transmission.pkl")

        # Continue with the rest of the code as before
        merged_df = merged_df.dropna()

        X = merged_df[[
            "dealer_type", "stock_type", "mileage", "price", "model_year",
            "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
        ]]
        y = merged_df["transmission_from_vin"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write("### Initial Model Evaluation:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        joblib.dump(model, "models/vehicle_transmission_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(list(X.columns), "models/original_columns.pkl")  # Save original columns as list

        st.success("Model trained and saved successfully.")

    except Exception as e:
        st.error(f"Error during feature engineering or model training: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    try:
        model = joblib.load('models/vehicle_transmission_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')  # Load encoders here
        le_transmission = joblib.load('models/le_transmission.pkl')  # Load label encoder here
        original_columns = joblib.load('models/original_columns.pkl')  # Load original columns here
        st.write("Model and files loaded successfully.")
    except Exception as e:
        st.error(f"Error loading files: {e}")

    st.subheader("Enter Vehicle Details:")

    dealer_type = st.selectbox("Dealer Type", merged_df['dealer_type'].unique())
    stock_type = st.selectbox("Stock Type", merged_df['stock_type'].unique())
    mileage = st.number_input("Mileage", min_value=0)
    price = st.number_input("Price", min_value=0)
    model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
    make = st.selectbox("Make", merged_df['make'].unique())
    model = st.selectbox("Model", merged_df['model'].unique())
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
            input_data = input_data.reindex(columns=original_columns, fill_value=0)

            for col, encoder in encoders.items():
                if col in input_data.columns:
                    input_data[col] = input_data[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else encoder.transform(['unknown'])[0])

            scaled_input = scaler.transform(input_data)

            prediction = model.predict(scaled_input)

            predicted_transmission = le_transmission.inverse_transform(prediction)

            st.write(f"### Predicted Transmission: {predicted_transmission[0]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Power BI dashboard link goes here.")

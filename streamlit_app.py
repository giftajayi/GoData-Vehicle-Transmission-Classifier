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
        return None

@st.cache_data
def optimize_dataframe(df):
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

merged_df = load_and_merge_data()
if merged_df is not None:
    merged_df = optimize_dataframe(merged_df)

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Helper function to encode categorical features
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
    if merged_df is not None:
        st.write("### Dataset Sample")
        st.write(merged_df.head())
        st.write("### Dataset Information")
        st.image("info1.jpeg", caption="Dataset Overview - Part 1")
        st.image("info2.jpeg", caption="Dataset Overview - Part 2")
        st.image("chart7.jpeg", caption="Transmission Distribution (Auto vs Manual)")
        st.image("chart2.png", caption="Price vs Mileage Scatter Plot")
        st.image("plt3.png", caption="Correlation Heatmap")
    else:
        st.error("No dataset available for analysis.")

# Feature Engineering and Model Training Section
elif section == "Feature Engineering and Model Training":
    st.title("🧑‍🔬 Feature Engineering and Model Training")
    if merged_df is not None:
        try:
            # Encoding features
            merged_df, encoders = encode_features(merged_df)

            le_transmission = LabelEncoder()
            merged_df["transmission_from_vin"] = le_transmission.fit_transform(merged_df["transmission_from_vin"])

            # Save encoders and label encoders
            joblib.dump(encoders, "models/encoders.pkl")
            joblib.dump(le_transmission, "models/le_transmission.pkl")

            # Prepare data
            X = merged_df[
                [
                    "dealer_type", "stock_type", "mileage", "price", "model_year",
                    "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
                ]
            ]
            y = merged_df["transmission_from_vin"]

            # Scale and split data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Train the model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            st.write("### Initial Model Evaluation:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            st.write("### Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Save model, scaler, and column info
            joblib.dump(model, "models/vehicle_transmission_model.pkl")
            joblib.dump(scaler, "models/scaler.pkl")
            joblib.dump(list(X.columns), "models/original_columns.pkl")

            st.success("Model trained and saved successfully.")
        except Exception as e:
            st.error(f"Error during model training: {e}")
    else:
        st.error("No dataset available for model training.")

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
    except Exception as e:
        st.error(f"Error loading files: {e}")
        model, scaler, encoders, le_transmission, original_columns = None, None, None, None, None

    if model:
        # Input for prediction
        st.subheader("Enter Vehicle Details:")
        input_data = {}
        for col in original_columns:
            if col in encoders:
                input_data[col] = st.selectbox(f"{col}", encoders[col].classes_)
            else:
                input_data[col] = st.number_input(f"{col}", value=0.0)

        if st.button("Generate Prediction"):
            try:
                # Prepare and scale input
                input_df = pd.DataFrame([input_data]).reindex(columns=original_columns, fill_value=0)
                for col, encoder in encoders.items():
                    if col in input_df.columns:
                        input_df[col] = encoder.transform(input_df[col].astype(str))
                scaled_input = scaler.transform(input_df)

                # Predict
                prediction = model.predict(scaled_input)
                predicted_transmission = le_transmission.inverse_transform(prediction)
                st.write(f"### Predicted Transmission: {predicted_transmission[0]}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

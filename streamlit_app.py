import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load and merge datasets
csv_urls = [
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data1.csv",
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data2.csv",
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data3.csv",
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data4.csv",
    "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/master/Cleaned_data5.csv",
]

@st.cache_data
def load_and_merge_data():
    dfs = [pd.read_csv(url) for url in csv_urls]
    merged = pd.concat(dfs, ignore_index=True)
    return merged

merged_df = load_and_merge_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Dashboard", "EDA", "ML Model", "ML Model Type", "Model Prediction", "Power BI Dashboard"]
)

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    # Collect input data from user
    mileage = st.number_input("Mileage (in km)", min_value=0)
    price = st.number_input("Price (in CAD)", min_value=0)
    model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
    fuel_type = st.selectbox("Fuel Type", merged_df['fuel_type_from_vin'].unique())
    certified = st.selectbox("Certified", ["Yes", "No"])
    price_changes = st.number_input("Price Changes", min_value=0)

    # Encode categorical inputs
    certified = 1 if certified == "Yes" else 0
    fuel_type_encoded = LabelEncoder().fit_transform([fuel_type])[0]

    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[mileage, price, model_year, fuel_type_encoded, certified, price_changes]],
                              columns=["mileage", "price", "model_year", "fuel_type_from_vin", "certified", "number_price_changes"])

    # Button to trigger prediction
    if st.button("Generate Prediction"):
        try:
            # Load the scaler and model
            scaler = joblib.load("scaler.pkl")
            model = joblib.load("vehicle_transmission_model.pkl")

            # Check that the input data columns match the expected model columns
            expected_columns = ["mileage", "price", "model_year", "fuel_type_from_vin", "certified", "number_price_changes"]
            if set(input_data.columns) != set(expected_columns):
                st.error(f"Input columns mismatch. Expected columns: {expected_columns}")
                st.stop()  # Stop execution here

            # Debug: show the input data
            st.write(f"Input data: {input_data}")

            # Scale input data
            input_data_scaled = scaler.transform(input_data)

            # Debug: show the scaled data
            st.write(f"Scaled input data: {input_data_scaled}")

            # Make the prediction
            prediction = model.predict(input_data_scaled)

            # Debug: show the prediction result
            st.write(f"Prediction raw result: {prediction}")

            # Display the prediction
            transmission_type = "Manual" if prediction[0] == 0 else "Automatic"
            st.write(f"### Predicted Transmission: **{transmission_type}**")

        except FileNotFoundError:
            st.error("Required model or scaler files not found. Please ensure 'scaler.pkl' and 'vehicle_transmission_model.pkl' are in place.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("""
    The dashboard provides insights and visualizations on transmission types, pricing trends, and more.
    """)

    # Link to Power BI Dashboard
    st.write("Click [here](https://app.powerbi.com/groups/me/reports/c9772dbc-0131-4e5a-a559-43a5c22874b3/ca237ccb0ae673ae960a?experience=power-bi) to view the Power BI dashboard.")

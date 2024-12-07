import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
import os

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
elif section == "Feature Engineering and Model Training":
    st.title("🧑‍🔬 Feature Engineering and Model Training")

    # Feature Engineering Steps
    st.subheader("🔧 Feature Engineering")

    st.write("""
    In this section, we apply transformations and preprocessing steps to prepare the data for training. 
    Feature engineering is critical as it impacts the model’s performance.
    """)

    try:
        # 1. Encoding target variable
        le_target = LabelEncoder()
        merged_df["transmission_from_vin"] = le_target.fit_transform(merged_df["transmission_from_vin"])

        # 2. Handling missing data
        merged_df = merged_df.dropna()

        # 3. Selecting features to use in the model
        X = merged_df[
            [
                "dealer_type", "stock_type", "mileage", "price", "model_year",
                "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
            ]
        ]
        y = merged_df["transmission_from_vin"]

        # 4. Encoding categorical features
        encoders = {}
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        # 5. Scaling numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save artifacts
        if not os.path.exists('models'):
            os.makedirs('models')

        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(le_target, "models/label_encoder.pkl")
        joblib.dump(encoders, "models/label_encoders.pkl")
        joblib.dump(X.columns, "models/original_columns.pkl")

        st.write("### Preprocessing completed: Features prepared for model training.")

    except Exception as e:
        st.error(f"Error during feature engineering: {e}")

    # Model Training Steps
    st.subheader("🏋️‍♂️ Model Training")
    st.write("""
    In this section, we will split the data into training and testing sets, train the RandomForestClassifier, 
    and evaluate its initial performance. 
    """)

    try:
        # 1. Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        st.write("### Data split into training and testing sets.")

        # 2. Training the RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        st.write("### Model training completed.")

        # Save the trained model
        joblib.dump(model, "models/vehicle_transmission_model.pkl")
        st.success("Model trained and saved successfully.")

    except Exception as e:
        st.error(f"Error during model training: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    def predict_transmission(input_data):
        model = joblib.load("models/vehicle_transmission_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        original_columns = joblib.load("models/original_columns.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")

        # Reindex to match the original columns used during training
        input_data = input_data.reindex(columns=original_columns, fill_value=0)
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        
        # Decode the prediction back to the original label
        return label_encoder.inverse_transform(prediction)

    st.subheader("Enter Vehicle Details:")
    mileage = st.number_input("Mileage (in km)", value=30000)
    price = st.number_input("Price (in CAD)", value=25000)
    model_year = st.number_input("Model Year", value=2020)
    number_price_changes = st.number_input("Number of Price Changes", value=3)
    certified = st.selectbox("Certified", ["Yes", "No"])
    fuel_type = st.selectbox("Fuel Type", ["Gas", "Diesel", "Electric", "Hybrid"])

    input_data = pd.DataFrame(
        [
            {
                "mileage": mileage,
                "price": price,
                "model_year": model_year,
                "number_price_changes": number_price_changes,
                "certified": 1 if certified == "Yes" else 0,
                "fuel_type_from_vin": fuel_type,
            }
        ]
    )

    if st.button("Generate Prediction"):
        try:
            prediction = predict_transmission(input_data)
            st.write(f"### Predicted Transmission: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Power BI dashboard link goes here.")

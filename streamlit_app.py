import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
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
        # 1. Encoding categorical variables using LabelEncoder
        le = LabelEncoder()
        merged_df["transmission_from_vin"] = le.fit_transform(merged_df["transmission_from_vin"])

        # 2. Handling missing data (if applicable)
        # We drop rows with missing values for simplicity. Alternatively, we could impute values.
        merged_df = merged_df.dropna()

        # 3. Selecting features to use in the model
        X = merged_df[[
            "dealer_type", "stock_type", "mileage", "price", "model_year",
            "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
        ]]
        
        # Target variable
        y = merged_df["transmission_from_vin"]

        # 4. Encoding categorical features in X (if any)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))

        # 5. Scaling numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

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

        # 3. Predicting and evaluating on the test set
        y_pred = model.predict(X_test)

        st.write("### Initial Model Evaluation:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Save the trained model
        joblib.dump(model, "models/vehicle_transmission_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")  # Save the scaler
        joblib.dump(le, "models/label_encoders.pkl")  # Save the label encoder
        joblib.dump(X.columns, "models/original_columns.pkl")  # Save original column names

        st.success("Model trained and saved successfully.")

    except Exception as e:
        st.error(f"Error during model training: {e}")
        
# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    def predict_transmission(input_data):
        model_path = "models/vehicle_transmission_model.pkl"
        
        if not os.path.exists(model_path):
            st.error("Model file not found. Please train the model first.")
            return None

        try:
            model = joblib.load(model_path)  # Load the trained model
            scaler = joblib.load("models/scaler.pkl")  # Load the scaler
            original_columns = joblib.load("models/original_columns.pkl")  # Load original column names
            label_encoder = joblib.load("models/label_encoders.pkl")  # Load the label encoder

            # Reindex input data to match the original columns
            input_data = input_data.reindex(columns=original_columns, fill_value=0)
            scaled_input = scaler.transform(input_data)  # Scale input data
            prediction = model.predict(scaled_input)  # Make prediction
            
            # Decode prediction
            return label_encoder.inverse_transform(prediction)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None

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
            if prediction is not None:
                st.write(
                    f"### Predicted Transmission: {prediction[0]}"
                )
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Power BI dashboard link goes here.")

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

# Feature Engineering and Model Training Section
if section == "Feature Engineering and Model Training":
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
        merged_df = merged_df.dropna()  # Drop rows with missing values

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

        # Save the trained model and necessary files
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

    # Load necessary files
    try:
        model = joblib.load('models/vehicle_transmission_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoders.pkl')
        original_columns = joblib.load('models/original_columns.pkl')
        st.write("Model and files loaded successfully.")
    except Exception as e:
        st.error(f"Error loading files: {e}")

    st.subheader("Enter Vehicle Details:")

    # User inputs for prediction (use features as specified)
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

    # Prepare the input data
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

    # Check the input data
    st.write("Input Data for Prediction:")
    st.write(input_data)

    # Make prediction when button is pressed
    if st.button("Generate Prediction"):
        try:
            # Ensure that the input data has the correct columns (align with original columns)
            input_data = input_data.reindex(columns=original_columns, fill_value=0)

            # Handle categorical features by encoding them
            for col in input_data.select_dtypes(include=['object']).columns:
                input_data[col] = label_encoder.fit_transform(input_data[col])

            # Scale the input data
            scaled_input = scaler.transform(input_data)

            # Make the prediction
            prediction = model.predict(scaled_input)

            # Decode the prediction label back to the original class
            predicted_transmission = label_encoder.inverse_transform(prediction)

            # Output the prediction
            st.write(f"### Predicted Transmission: {predicted_transmission[0]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Power BI dashboard link goes here.")

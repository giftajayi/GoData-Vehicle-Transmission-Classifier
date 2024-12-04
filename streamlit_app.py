import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
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

# Dashboard Section
if section == "Dashboard":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    This project uses machine learning to classify vehicles as either **Manual** or **Automatic**.
    The goal is to help optimize inventory management, marketing, and sales strategies for Go Auto by predicting 
    the transmission type of vehicles in their listings.
    """)

# Exploratory Data Analysis (EDA) Section
elif section == "EDA":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.write("### First 5 Rows of the Dataset:")
    st.dataframe(merged_df.head(5))
    st.write("### Dataset Statistics:")
    st.write(merged_df.describe())
    st.write("### Distribution of Transmission Types:")
    st.bar_chart(merged_df['transmission_from_vin'].value_counts())
    st.write("### Correlation Heatmap:")
    numeric_df = merged_df.select_dtypes(include=['float64', 'int64'])
    st.write(numeric_df.corr())

# ML Model Section
elif section == "ML Model":
    st.title("🏋️ Model Training & Evaluation")
    try:
        merged_df.dropna(subset=['transmission_from_vin'], inplace=True)
        le = LabelEncoder()
        merged_df['transmission_encoded'] = le.fit_transform(merged_df['transmission_from_vin'])
        features = ["mileage", "price", "model_year", "fuel_type_from_vin", "certified", "number_price_changes"]
        X = merged_df[features]
        y = merged_df['transmission_encoded']
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.pkl")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        smote = SMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        model = RandomForestClassifier()
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        st.write("### Accuracy:", accuracy_score(y_test, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=['Manual', 'Automatic']))
        st.write("### Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        joblib.dump(model, "vehicle_transmission_model.pkl")
        st.success("Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Model training error: {e}")

# ML Model Type Section
elif section == "ML Model Type":
    st.title("🧠 ML Model Type")
    st.write("""
    For this classification task, we are using a **Random Forest Classifier**. It is an ensemble learning method
    that works by creating multiple decision trees and combining their predictions. This model is effective for
    classification problems and helps in handling complex data with multiple features.
    """)

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    # Collect input data from user
    mileage = st.number_input("Mileage (in km)", min_value=0, value=50000)
    price = st.number_input("Price (in CAD)", min_value=0, value=25000)
    model_year = st.number_input("Model Year", min_value=2000, max_value=2024, value=2020)
    fuel_type = st.selectbox("Fuel Type", merged_df['fuel_type_from_vin'].unique())
    certified = st.selectbox("Certified", ["Yes", "No"])
    price_changes = st.number_input("Price Changes", min_value=0, value=2)

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
                return

            # Scale input data
            input_data_scaled = scaler.transform(input_data)

            # Make the prediction
            prediction = model.predict(input_data_scaled)
            transmission_type = "Manual" if prediction[0] == 0 else "Automatic"
            st.write(f"### Predicted Transmission: **{transmission_type}**")

        except FileNotFoundError:
            st.error("Required model or scaler files not found. Please ensure 'scaler.pkl' and 'vehicle_transmission_model.pkl' are in place.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    power_bi_url = "https://app.powerbi.com/view?r=YOUR_POWER_BI_EMBED_URL"
    iframe_code = f'<iframe width="100%" height="600" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>'
    st.markdown(iframe_code, unsafe_allow_html=True)

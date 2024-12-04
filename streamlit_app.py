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
    ["Dashboard", "EDA", "ML Model", "ML Model Type", "Model Prediction"]
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
    corr = merged_df.corr()
    st.write(corr)

# ML Model Section
elif section == "ML Model":
    st.title("🏋️ Model Training & Evaluation")
    
    try:
        # Data Preprocessing
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

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Handle class imbalance
        smote = SMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Train Model
        model = RandomForestClassifier()
        model.fit(X_train_res, y_train_res)

        # Evaluation
        y_pred = model.predict(X_test)
        st.write("### Accuracy:", accuracy_score(y_test, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=['Manual', 'Automatic']))
        st.write("### Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        # Save the model
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
    classification problems and helps in handling complex data with multiple features, such as vehicle characteristics.
    """)
    st.write("### Why Random Forest?")
    st.write("""
    - **Robustness**: It handles large datasets with higher dimensionality well.
    - **Non-linearity**: It captures complex relationships between variables.
    - **Feature importance**: It can rank features based on their importance.
    """)

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    st.write("""
    Use the form below to predict whether a vehicle is **Manual** or **Automatic** based on its characteristics.
    Select the values for the following features:
    """)

    # Interactive form to input values
    mileage = st.number_input("Mileage (in km)", min_value=0, value=50000)
    price = st.number_input("Price (in CAD)", min_value=0, value=25000)
    model_year = st.number_input("Model Year", min_value=2000, max_value=2024, value=2020)
    fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
    certified = st.selectbox("Certified", ["Yes", "No"])
    price_changes = st.number_input("Price Changes", min_value=0, value=2)

    # Preprocessing for prediction
    fuel_type = le.transform([fuel_type])[0]
    certified = 1 if certified == "Yes" else 0

    # Prepare the features for prediction
    input_data = pd.DataFrame([[mileage, price, model_year, fuel_type, certified, price_changes]],
                              columns=["mileage", "price", "model_year", "fuel_type_from_vin", "certified", "number_price_changes"])

    input_data_scaled = scaler.transform(input_data)

    # Load model and predict
    model = joblib.load("vehicle_transmission_model.pkl")
    prediction = model.predict(input_data_scaled)

    transmission_type = "Manual" if prediction[0] == 0 else "Automatic"
    st.write(f"### Predicted Transmission: **{transmission_type}**")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("""
    The dashboard provides insights and visualizations on transmission types, pricing trends, and more.
    """)

    # Link to Power BI Dashboard
    st.write("Click [here](https://app.powerbi.com/groups/me/reports/c9772dbc-0131-4e5a-a559-43a5c22874b3/ca237ccb0ae673ae960a?experience=power-bi) to view the Power BI dashboard.")

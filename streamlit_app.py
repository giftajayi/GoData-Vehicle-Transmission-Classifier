import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

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
    ["Dashboard", "EDA", "ML Model", "Model Prediction", "Power BI Dashboard"]
)

# Dashboard Section
if section == "Dashboard":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    The primary objective of this project is to develop a machine learning model that can reliably predict whether a vehicle has an automatic or manual transmission...
    """)

# EDA Section
elif section == "EDA":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    # Display preloaded images as examples

# ML Model Section
elif section == "ML Model":
    st.title("🏋️ Model Training & Hyperparameter Tuning")
    try:
        # Prepare dataset
        merged_df.dropna(subset=['transmission_from_vin'], inplace=True)
        le = LabelEncoder()
        merged_df['transmission_encoded'] = le.fit_transform(merged_df['transmission_from_vin'])

        features = ["dealer_type", "stock_type", "mileage", "price", "model_year", "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"]
        X = merged_df[features]
        y = merged_df['transmission_encoded']

        # Encode categorical variables
        X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.pkl")

        # Train-test split and balancing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        smote = SMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_res, y_train_res)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        st.write("### Best Hyperparameters:", grid_search.best_params_)
        st.write("### Accuracy:", accuracy_score(y_test, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=['Manual', 'Automatic']))
        st.write("### Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        joblib.dump(best_model, "vehicle_transmission_model.pkl")
        st.success("Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Model training error: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("vehicle_transmission_model.pkl")
        st.write("Model loaded successfully.")

        dealer_type = st.selectbox("Dealer Type", merged_df['dealer_type'].unique())
        stock_type = st.selectbox("Stock Type", merged_df['stock_type'].unique())
        mileage = st.number_input("Mileage", min_value=0)
        price = st.number_input("Price", min_value=0)
        model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
        make = st.selectbox("Make", merged_df['make'].unique())
        model_input = st.selectbox("Model", merged_df['model'].unique())
        certified = st.radio("Certified", ["Yes", "No"])
        fuel_type = st.selectbox("Fuel Type", merged_df['fuel_type_from_vin'].unique())
        price_changes = st.number_input("Number of Price Changes", min_value=0)

        # Create input dataframe for prediction
        input_data = pd.DataFrame({
            'dealer_type': [dealer_type],
            'stock_type': [stock_type],
            'mileage': [mileage],
            'price': [price],
            'model_year': [model_year],
            'make': [make],
            'model': [model_input],
            'certified': [1 if certified == "Yes" else 0],
            'fuel_type_from_vin': [fuel_type],
            'number_price_changes': [price_changes]
        })

        # Encoding categorical features for prediction
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Scale input features
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        transmission = le.inverse_transform(prediction)[0]

        st.write(f"Predicted Transmission Type: {transmission}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Power BI Section (not implemented here)
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Power BI dashboard can be embedded here.")

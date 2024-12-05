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

# Load and Merge Datasets
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
    return pd.concat(dfs, ignore_index=True)

merged_df = load_and_merge_data()

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Dashboard", "EDA", "ML Model", "Model Prediction", "Power BI Dashboard"])

# Dashboard Section
if section == "Dashboard":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    The primary objective of this project is to develop a machine learning model to predict a vehicle's transmission type, enhancing Go Auto’s decision-making and marketing strategies.
    """)

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

# ML Model Section
elif section == "ML Model":
    st.title("🏋️ Model Training & Hyperparameter Tuning")

    def train_model():
        merged_df.dropna(subset=['transmission_from_vin'], inplace=True)
        le = LabelEncoder()
        merged_df['transmission_encoded'] = le.fit_transform(merged_df['transmission_from_vin'])

        features = ["dealer_type", "stock_type", "mileage", "price", "model_year", "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"]
        X = merged_df[features]
        y = merged_df['transmission_encoded']

        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.pkl")

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        smote = SMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_res, y_train_res)
        
        return grid_search, X_test, y_test, features

    try:
        grid_search, X_test, y_test, features = train_model()
        y_pred = grid_search.best_estimator_.predict(X_test)
        st.write("### Best Hyperparameters:", grid_search.best_params_)
        st.write("### Accuracy:", accuracy_score(y_test, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.write("### Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        joblib.dump(grid_search.best_estimator_, "vehicle_transmission_model.pkl")
    except Exception as e:
        st.error(f"Model training error: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    def predict_transmission(input_data):
        model = joblib.load("vehicle_transmission_model.pkl")
        scaler = joblib.load("scaler.pkl")
        scaled_input = scaler.transform(input_data)
        return model.predict(scaled_input)

    # Example feature input for user testing
    st.subheader("Enter Vehicle Details:")
    input_data = pd.DataFrame([{
        "dealer_type": "Used", 
        "stock_type": "Certified", 
        "mileage": 30000, 
        "price": 25000, 
        "model_year": 2020, 
        "make": "Toyota", 
        "model": "Corolla", 
        "certified": 1, 
        "fuel_type_from_vin": "Gasoline", 
        "number_price_changes": 3
    }])  # Example data; integrate with user inputs later.

    # Preprocess input
    input_data = pd.get_dummies(input_data.reindex(columns=features, fill_value=0))

    if st.button("Generate Prediction"):
        try:
            prediction = predict_transmission(input_data)
            st.write(f"### Predicted Transmission: {'Automatic' if prediction[0] == 1 else 'Manual'}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

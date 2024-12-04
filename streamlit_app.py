import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
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
    ["Dashboard", "EDA", "ML Model", "Model Prediction", "Power BI Dashboard"]
)

# Dashboard Section
if section == "Dashboard":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    This project uses machine learning to classify vehicles as either **Manual** or **Automatic**.
    """)

# Exploratory Data Analysis (EDA) Section
elif section == "EDA":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.write("""
    Visualizations help us understand patterns, distributions, and relationships within the dataset.
    """)

# ML Model Section
elif section == "ML Model":
    st.title("🏋️ Model Training & Evaluation")
    try:
        # Data preprocessing
        merged_df.dropna(subset=['transmission_from_vin'], inplace=True)
        le = LabelEncoder()
        merged_df['transmission_encoded'] = le.fit_transform(merged_df['transmission_from_vin'])
        
        features = ["dealer_type", "stock_type", "mileage", "price", "model_year", 
                    "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"]
        X = merged_df[features]
        y = merged_df['transmission_encoded']
        
        # Encoding categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Feature selection using Chi-Square
        chi_selector = SelectKBest(chi2, k=5)
        X_selected = chi_selector.fit_transform(X, y)
        st.write("### Selected Features after Chi-Square:", list(X.columns[chi_selector.get_support()]))

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        joblib.dump(scaler, "scaler.pkl")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        
        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_res, y_train_res)
        best_model = grid_search.best_estimator_
        
        # Model evaluation
        y_pred = best_model.predict(X_test)
        st.write("### Best Parameters:", grid_search.best_params_)
        st.write("### Accuracy:", accuracy_score(y_test, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=['Manual', 'Automatic']))
        st.write("### Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        
        # Save the model
        joblib.dump(best_model, "vehicle_transmission_model.pkl")
        st.success("Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Model training error: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")
    mileage = st.number_input("Mileage (in km)", min_value=0)
    price = st.number_input("Price (in CAD)", min_value=0)
    model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
    fuel_type = st.selectbox("Fuel Type", merged_df['fuel_type_from_vin'].unique())
    certified = st.selectbox("Certified", ["Yes", "No"])
    price_changes = st.number_input("Price Changes", min_value=0)

    certified = 1 if certified == "Yes" else 0
    fuel_type_encoded = LabelEncoder().fit_transform([fuel_type])[0]
    input_data = pd.DataFrame([[mileage, price, model_year, fuel_type_encoded, certified, price_changes]],
                              columns=["mileage", "price", "model_year", "fuel_type_from_vin", "certified", "number_price_changes"])

    if st.button("Generate Prediction"):
        try:
            scaler = joblib.load("scaler.pkl")
            model = joblib.load("vehicle_transmission_model.pkl")
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)
            transmission_type = "Manual" if prediction[0] == 0 else "Automatic"
            st.write(f"### Predicted Transmission: **{transmission_type}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("""
    The dashboard provides insights and visualizations on transmission types, pricing trends, and more.
    """)

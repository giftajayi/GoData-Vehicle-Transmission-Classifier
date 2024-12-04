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
        
        # Encoding categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.pkl")
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Apply SMOTE to handle class imbalance only on the training data
        smote = SMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Train the Random Forest model with class weights balanced
        model = RandomForestClassifier(class_weight='balanced')
        model.fit(X_train_res, y_train_res)
        
        # Evaluate the model
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
    st.write("Click [here](https://app.powerbi.com/groups/me/reports/c9772dbc-0131-4e5a-a559-43a5c22874b3/ca237ccb0ae673ae960a?experience=power-bi) to view the Power BI dashboard.")

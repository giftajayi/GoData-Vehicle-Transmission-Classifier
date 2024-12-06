import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    dfs = [pd.read_csv(url) for url in csv_urls]
    return pd.concat(dfs, ignore_index=True)

merged_df = load_and_merge_data()

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

    st.subheader("🔧 Feature Engineering")
    try:
        # Encode the target variable
        le = LabelEncoder()
        merged_df["transmission_from_vin"] = le.fit_transform(
            merged_df["transmission_from_vin"]
        )

        # Drop missing data
        merged_df = merged_df.dropna()

        # Select features and target variable
        feature_columns = [
            "dealer_type",
            "stock_type",
            "mileage",
            "price",
            "model_year",
            "make",
            "model",
            "certified",
            "fuel_type_from_vin",
            "number_price_changes",
        ]
        X = merged_df[feature_columns]
        y = merged_df["transmission_from_vin"]

        # Encode categorical features and scale numerical features
        X = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler and feature columns for future use
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(X.columns.tolist(), "original_columns.pkl")

        st.write("### Preprocessing completed successfully.")
    except Exception as e:
        st.error(f"Error during feature engineering: {e}")

    st.subheader("🏋️‍♂️ Model Training and Evaluation")
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train the model with balanced class weights
        model = RandomForestClassifier(
            random_state=42, class_weight="balanced", n_estimators=100, max_depth=10
        )
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, "vehicle_transmission_model.pkl")

        # Evaluate the model
        y_pred = model.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Display evaluation results
        st.write(f"### Model Accuracy: {acc_score:.4f}")
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix:")
        st.dataframe(
            pd.DataFrame(conf_matrix, index=le.classes_, columns=le.classes_)
        )

        st.success("Model training and evaluation completed successfully.")
    except Exception as e:
        st.error(f"Error during model training: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    def predict_transmission(input_data):
        model = joblib.load("vehicle_transmission_model.pkl")
        scaler = joblib.load("scaler.pkl")
        original_columns = joblib.load("original_columns.pkl")
        input_data = input_data.reindex(columns=original_columns, fill_value=0)
        scaled_input = scaler.transform(input_data)
        return model.predict(scaled_input)

    st.subheader("Enter Vehicle Details:")

    make_model_dict = {
        "Chrysler": ["Fifth Avenue", "300", "Pacifica"],
        "Cadillac": ["DeVille", "Escalade", "CTS"],
        "Volkswagen": ["Cabriolet", "Jetta", "Passat"],
    }

    make = st.selectbox("Make", list(make_model_dict.keys()))
    model = st.selectbox("Model", make_model_dict[make])
    dealer_type = st.selectbox("Dealer Type", ["I", "F"])
    stock_type = st.selectbox("Stock Type", ["Used", "New"])
    mileage = st.number_input("Mileage (in km)", value=30000)
    price = st.number_input("Price (in CAD)", value=25000)
    model_year = st.number_input("Model Year", value=2020)
    certified = st.selectbox("Certified", ["Yes", "No"])
    fuel_type = st.selectbox("Fuel Type", ["Gas", "Diesel", "CNG", "Electric", "Hybrid"])
    number_price_changes = st.number_input("Number of Price Changes", value=3)

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
                "number_price_changes": number_price_changes,
            }
        ]
    )

    if st.button("Generate Prediction"):
        try:
            prediction = predict_transmission(input_data)
            st.write(
                f"### Predicted Transmission: {'Automatic' if prediction[0] == 1 else 'Manual'}"
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write(
        """
        The dashboard provides insights and visualizations on transmission types, pricing trends, and more.
        """
    )
    st.write(
        "Click [here](https://app.powerbi.com/groups/me/reports/c9772dbc-0131-4e5a-a559-43a5c22874b3/ca237ccb0ae673ae960a?experience=power-bi) to view the Power BI dashboard."
    )

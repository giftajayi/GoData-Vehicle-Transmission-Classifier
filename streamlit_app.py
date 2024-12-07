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

    @st.cache_data
    def preprocess_data(df):
        feature_columns = [
            "dealer_type",
            "stock_type",
            "mileage",
            "price",
            "model_year",
            "certified",
            "fuel_type_from_vin",
            "number_price_changes",
        ]
        # One-hot encoding for categorical features
        X = pd.get_dummies(df[feature_columns], drop_first=True)

        # Encode the target variable (Manual -> 0, Automatic -> 1)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["transmission_from_vin"])  # Encode target labels
        joblib.dump(label_encoder, "label_encoder.pkl")  # Save the encoder for future predictions

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(X.columns.tolist(), "original_columns.pkl")

        return train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    try:
        # Preprocess the data
        X_train, X_test, y_train, y_test = preprocess_data(merged_df)

        # SMOTE for balancing the target classes
        smote = SMOTE(sampling_strategy={0: 3000, 1: 2000}, random_state=42)  # Adjust numbers as needed
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Check resampled class distribution
        st.write("### Resampled Class Distribution:")
        st.write(pd.Series(y_train_resampled).value_counts())

        # Model Training
        model = RandomForestClassifier(
            random_state=42,
            class_weight={0: 3, 1: 1},  # Adjust weights for Manual (0) and Automatic (1)
            n_estimators=100,
            max_depth=10,
        )
        model.fit(X_train_resampled, y_train_resampled)

        # Save the model
        joblib.dump(model, "vehicle_transmission_model.pkl")

        # Evaluation
        y_pred = model.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {acc_score:.4f}")
        st.write("### Classification Report:")
        label_encoder = joblib.load("label_encoder.pkl")
        st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        st.write("### Confusion Matrix:")
        st.dataframe(
            pd.DataFrame(
                confusion_matrix(y_test, y_pred),
                columns=["Predicted Manual", "Predicted Automatic"],
                index=["Actual Manual", "Actual Automatic"],
            )
        )
    except Exception as e:
        st.error(f"Error during model training: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    def predict_transmission(input_data):
        model = joblib.load("vehicle_transmission_model.pkl")
        scaler = joblib.load("scaler.pkl")
        original_columns = joblib.load("original_columns.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

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
            st.write(
                f"### Predicted Transmission: {prediction[0]}"
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Power BI dashboard link goes here.")

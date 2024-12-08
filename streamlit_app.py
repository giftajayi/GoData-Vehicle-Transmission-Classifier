
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

# Helper function to encode categorical features using LabelEncoder
def encode_features(df, encoders=None):
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in df.select_dtypes(include=['object']).columns}
    
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    
    return df, encoders

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "EDA",
        "Model Building",
        "Model Prediction",
        "Power BI Dashboard",
    ],
)

# Dashboard Section
# Check if the images exist before attempting to display
if os.path.exists("logo1.png") and os.path.exists("logo2.png"):
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image("logo1.png", width=50)
    with col2:
        st.image("logo2.png", width=50)
else:
    st.error("Logos not found. Please check the file paths.")

# Dashboard Title
    st.title("🚗 Vehicle Transmission Classifier")

    # Project description
    st.write(
        """
        The primary objective of this project is to develop a machine learning model that can reliably predict whether a vehicle has an automatic or manual transmission. 
        By leveraging this model, Go Auto aims to enhance its decision-making processes, streamline inventory classification, and target marketing efforts more effectively. 
        A successful model would not only improve operational efficiency but also provide valuable insights into consumer preferences, helping dealerships better align their offerings with market demand. 
        The ability to accurately identify the transmission type can contribute significantly to improving customer experiences and boosting sales.
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

# Model Building Section
if section == "Model Building":  # Corrected section name by removing extra spaces
    st.title("🧑‍🔬  Model Building ")
    st.write(
        """ The model was built to evaluate the performance of various machine learning classifiers in predicting the target variable. The process began by selecting a diverse range of models, including Logistic Regression, K-Nearest Neighbors, Naive Bayes, Support Vector Machines, Random Forest, Decision Tree, and XGBoost. To handle missing values, a `SimpleImputer` was employed to replace them with a constant (0), ensuring consistency across all folds of the training data. Each model was incorporated into a pipeline alongside the imputer, streamlining the preprocessing and training stages.
        To ensure robust evaluation, 5-fold cross-validation was conducted for each pipeline. This method split the data into training and testing subsets in multiple iterations, calculating the accuracy for each fold. The mean and standard deviation of these accuracy scores provided insights into the performance and stability of the models. The results were stored systematically, allowing for easy comparison and enabling the selection of the most effective classifier for the dataset. This approach ensured fairness in evaluation and enhanced the reliability of the chosen model.
        """
    )

    try:
        merged_df, encoders = encode_features(merged_df)

        le_transmission = LabelEncoder()
        merged_df["transmission_from_vin"] = le_transmission.fit_transform(merged_df["transmission_from_vin"])

        # Save the encoders and label encoder for later use
        joblib.dump(encoders, "models/encoders.pkl")
        joblib.dump(le_transmission, "models/le_transmission.pkl")

        merged_df = merged_df.dropna()

        X = merged_df[[  # Features
            "dealer_type", "stock_type", "mileage", "price", "model_year",
            "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
        ]]
        y = merged_df["transmission_from_vin"]  # Target

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write("### Initial Model Evaluation:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Save models and required files
        joblib.dump(model, "models/vehicle_transmission_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(list(X.columns), "models/original_columns.pkl")

        st.success("Model trained and saved successfully.")

    except Exception as e:
        st.error(f"Error during feature engineering or model training: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")
    try:
        # Load model and related files
        model = joblib.load('models/vehicle_transmission_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')
        le_transmission = joblib.load('models/le_transmission.pkl')
        original_columns = joblib.load('models/original_columns.pkl')
    except Exception as e:
        st.error(f"Error loading files: {e}")
        model, scaler, encoders, le_transmission, original_columns = None, None, None, None, None

    if model:
        # Input for prediction
        st.subheader("Enter Vehicle Details:")

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
    
        input_data = pd.DataFrame(
            [
                {
                    "dealer_type": dealer_type,
                    "stock_type": stock_type,
                    "mileage": mileage,
                    "price": price,
                    "model_year": model_year,
                    "make": make,
                    "model": model_input,
                    "certified": 1 if certified == "Yes" else 0,
                    "fuel_type_from_vin": fuel_type,
                    "number_price_changes": price_changes,
                }
            ]
        )
    
        st.write("Input Data for Prediction:")
        st.write(input_data)

        if st.button("Generate Prediction"):
            try:
                # Reindex input data to match the original columns
                input_df = input_data.reindex(columns=original_columns, fill_value=0)

                # Apply the encoders to categorical features
                for col, encoder in encoders.items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = encoder.transform(input_df[col].astype(str))
                        except KeyError:
                            # If category not seen during training, use a default encoding (e.g., -1 or the most frequent class)
                            input_df[col] = encoder.transform([input_df[col].mode()[0]])[0]

                # Scale the input data
                scaled_input = scaler.transform(input_df)

                # Predict transmission type
                prediction = model.predict(scaled_input)

                # Map predictions to the corresponding transmission type (Manual/Automatic)
                transmission_mapping = {0: "Automatic", 1: "Manual"}
                predicted_transmission = transmission_mapping.get(prediction[0], "Unknown")

                st.write(f"### Predicted Transmission: {predicted_transmission}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Click [here](https://app.powerbi.com/groups/me/reports/c9772dbc-0131-4e5a-a559-43a5c22874b3/ca237ccb0ae673ae960a?experience=power-bi) to view the Power BI dashboard.")


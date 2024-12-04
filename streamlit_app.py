import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

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
    The primary objective of this project is to develop a machine learning model that can reliably predict whether a vehicle has an automatic or manual transmission. By leveraging this model, Go Auto aims to enhance its decision-making processes, streamline inventory classification, and target marketing efforts more effectively. A successful model would not only improve operational efficiency but also provide valuable insights into consumer preferences, helping dealerships better align their offerings with market demand. The ability to accurately identify the transmission type can contribute significantly to improving customer experiences and boosting sales.
    """)

# Exploratory Data Analysis (EDA) Section
elif section == "EDA":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.write("""
    Visualizations help us understand patterns, distributions, and relationships within the dataset.
    """)

    st.subheader("1️⃣ Transmission Type Distribution")
    st.image("chart1.png", caption="Proportion of Automatic vs Manual Transmissions")
    st.write("This pie chart shows the proportion of vehicles with automatic versus manual transmissions.")

    st.subheader("2️⃣ Price vs Mileage Scatter Plot")
    st.image("chart2.png", caption="Price vs Mileage for Different Vehicles")
    st.write("This scatter plot highlights how vehicle price relates to mileage, offering insights into pricing patterns.")

    st.subheader("3️⃣ Correlation Heatmap")
    st.image("plt3.png", caption="Correlation Among Dataset Features")
    st.write("This heatmap illustrates the strength of correlations between key features in the dataset.")

    st.subheader("4️⃣ Model Year Distribution")
    st.image("plt4.png", caption="Distribution of Vehicles by Model Year")
    st.write("This bar chart shows how vehicle models are distributed across different production years.")

    st.subheader("5️⃣ Price Distribution by Fuel Type")
    st.image("chart5.png", caption="Price Variation Across Fuel Types")
    st.write("This plot compares price distributions across different fuel types, helping identify trends by fuel preference.")

    st.subheader("6️⃣ Mileage Boxplot by Transmission Type")
    st.image("plt6.png", caption="Mileage Distribution for Automatic and Manual Transmissions")
    st.write("This boxplot presents mileage variability for both automatic and manual transmissions.")

    st.subheader("7️⃣ Price vs Model Year Trend")
    st.image("plt7.png", caption="Average Price Trends by Model Year")
    st.write("This line chart tracks average vehicle prices over different model years, showing depreciation trends.")

    st.subheader("8️⃣ Make Popularity Countplot")
    st.image("chart6.png", caption="Frequency of Vehicle Makes in the Dataset")
    st.write("This count plot displays the frequency of various vehicle makes, indicating market preferences.")

# ML Model Section
elif section == "ML Model":
    st.title("🏋️ Model Training & Hyperparameter Tuning")
    try:
        merged_df.dropna(subset=['transmission_from_vin'], inplace=True)
        le = LabelEncoder()
        merged_df['transmission_encoded'] = le.fit_transform(merged_df['transmission_from_vin'])

        features = ["dealer_type", "stock_type", "mileage", "price", "model_year", "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"]
        X = merged_df[features]
        y = merged_df['transmission_encoded']

        # Encoding categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.pkl")  # Ensure it's saved to the correct path

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        smote = SMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
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

        # Save the tuned model
        joblib.dump(best_model, "vehicle_transmission_model.pkl")  # Save to the correct path
        st.success("Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Model training error: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")
    
    # User inputs for prediction
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

    # Encode 'Certified' feature (Yes=1, No=0)
    certified_encoded = 1 if certified == "Yes" else 0

    # Load LabelEncoders for each categorical feature
    le_dict = {
        'dealer_type': LabelEncoder().fit(merged_df['dealer_type']),
        'stock_type': LabelEncoder().fit(merged_df['stock_type']),
        'make': LabelEncoder().fit(merged_df['make']),
        'model': LabelEncoder().fit(merged_df['model']),
        'fuel_type_from_vin': LabelEncoder().fit(merged_df['fuel_type_from_vin'])
    }

    # Encode user inputs based on pre-trained label encoders
    dealer_type_encoded = le_dict['dealer_type'].transform([dealer_type])[0]
    stock_type_encoded = le_dict['stock_type'].transform([stock_type])[0]
    make_encoded = le_dict['make'].transform([make])[0]
    model_encoded = le_dict['model'].transform([model])[0]
    fuel_type_encoded = le_dict['fuel_type_from_vin'].transform([fuel_type])[0]

    # Load the saved model and scaler
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("vehicle_transmission_model.pkl")

        # Prepare input features for prediction
        input_data = pd.DataFrame({
            'dealer_type': [dealer_type_encoded],
            'stock_type': [stock_type_encoded],
            'mileage': [mileage],
            'price': [price],
            'model_year': [model_year],
            'make': [make_encoded],
            'model': [model_encoded],
            'certified': [certified_encoded],
            'fuel_type_from_vin': [fuel_type_encoded],
            'number_price_changes': [price_changes]
        })

        # Scale input features
        input_scaled = scaler.transform(input_data)

        # Predict the transmission type
        prediction = model.predict(input_scaled)
        transmission = le.inverse_transform(prediction)[0]

        st.write(f"Predicted Transmission Type: {transmission}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Power BI Section (not implemented here)
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Power BI dashboard can be embedded here.")

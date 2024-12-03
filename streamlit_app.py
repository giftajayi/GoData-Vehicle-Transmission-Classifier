import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Define paths to multiple datasets
csv_files = {
    "Main Dataset": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data1.csv",
    "Dataset 2": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data2.csv",
    "Dataset 3": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data3.csv",
    "Dataset 4": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data4.csv",
    "Dataset 5": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data5.csv",
}

# Load datasets
dataframes = {}
for name, path in csv_files.items():
    try:
        dataframes[name] = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load {name}: {e}")

# Concatenate datasets vertically
try:
    merged_df = dataframes["Main Dataset"]
    for name, df in dataframes.items():
        if name != "Main Dataset":
            merged_df = pd.concat([merged_df, df], ignore_index=True)
except Exception as e:
    st.error(f"An error occurred during merging: {e}")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Introduction", "Dataset Overview", "Visualization", "Feature Engineering and Model Training", "Model Evaluation", "Power BI Dashboard"],
    key="section_radio"
)

# Introduction Section
if section == "Introduction":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    The primary objective of this project is to develop a machine learning model that can reliably predict whether a vehicle has an automatic or manual transmission. By leveraging this model, Go Auto aims to enhance its decision-making processes, streamline inventory classification, and target marketing efforts more effectively. A successful model would not only improve operational efficiency but also provide valuable insights into consumer preferences, helping dealerships better align their offerings with market demand. The ability to accurately identify the transmission type can contribute significantly to improving customer experiences and boosting sales.

The Go Auto business intelligence team has provided a comprehensive dataset consisting of over 140,000 vehicle listings from various dealerships in Edmonton. This dataset contains a wide array of vehicle-related details, including attributes such as mileage, price, model year, make, and more. The GoData team’s challenge is to sift through this data and identify the most relevant features that will enable the development of an accurate classification model. With the goal of predicting transmission types, the team will focus on data preprocessing, feature selection, and machine learning techniques to ensure the model performs well across diverse vehicle listings. By doing so, the team aims to create a tool that offers actionable insights and supports Go Auto’s operational and strategic objectives.
    """)

# Dataset Overview Section
elif section == "Dataset Overview":
    st.title("📊 Dataset Overview")
    main_columns = [
        "vehicle_id", "dealer_type", "stock_type", "model_year", "model", "make",
        "price", "mileage", "transmission_type", "certified", "transmission_from_vin",
        "fuel_type_from_vin", "number_price_changes"
    ]
    available_columns = [col for col in main_columns if col in merged_df.columns]

    if available_columns:
        st.write("### First 5 Rows of the Dataset:")
        st.dataframe(merged_df[available_columns].head(5))
    else:
        st.warning("Main columns not found in the dataset.")

    st.write("### General Statistics:")
    st.write(merged_df.describe())

# Visualization Section
elif section == "Data Visualization":
st.title("📊 Data Visualizations")
st.write("""
These charts help uncover patterns 
and relationships between variables in the vehicle transmission dataset.
""")

    st.subheader("1️⃣ Transmission Type Distribution")
    st.image("chart1.png", caption="Proportion of Automatic vs Manual Transmissions", use_column_width=True)
    
    st.subheader("2️⃣ Price vs Mileage Scatter Plot")
    st.image("chart2.png", caption="Price vs Mileage for Different Vehicles", use_column_width=True)
    
    st.subheader("3️⃣ Correlation Heatmap")
    st.image("plt3.png", caption="Correlation Among Dataset Features", use_column_width=True)
    
    st.subheader("4️⃣ Model Year Distribution")
    st.image("plt4.png", caption="Distribution of Vehicles by Model Year", use_column_width=True)
    
    st.subheader("5️⃣ Price Distribution by Fuel Type")
    st.image("chart5.png", caption="Price Variation Across Fuel Types", use_column_width=True)
    
    st.subheader("6️⃣ Mileage Boxplot by Transmission Type")
    st.image("plt6.png", caption="Mileage Distribution for Automatic and Manual Transmissions", use_column_width=True)
    
    st.subheader("7️⃣ Price vs Model Year Trend")
    st.image("plt7.png", caption="Average Price Trends by Model Year", use_column_width=True)
    
    st.subheader("8️⃣ Make Popularity Countplot")
    st.image("chart6.png", caption="Frequency of Vehicle Makes in the Dataset", use_column_width=True)

# Feature Engineering and Model Training Section
elif section == "Feature Engineering and Model Training":
    st.title("🧑‍🔬 Feature Engineering and Model Training")

    # Feature Engineering Steps
    st.subheader("🔧 Feature Engineering")

    st.write("""
    In this section, we apply transformations and preprocessing steps to prepare the data for training. 
    Feature engineering is critical as it impacts the model’s performance.
    """)

    try:
        # 1. Encoding categorical variables using LabelEncoder
        le = LabelEncoder()
        merged_df["transmission_from_vin"] = le.fit_transform(merged_df["transmission_from_vin"])

        # 2. Handling missing data (if applicable)
        # We drop rows with missing values for simplicity. Alternatively, we could impute values.
        merged_df = merged_df.dropna()

        # 3. Selecting features to use in the model
        X = merged_df[[
            "dealer_type", "stock_type", "mileage", "price", "model_year",
            "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
        ]]
        
        # Target variable
        y = merged_df["transmission_from_vin"]

        # 4. Encoding categorical features in X (if any)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))

        # 5. Scaling numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.write("### Preprocessing completed: Features prepared for model training.")

    except Exception as e:
        st.error(f"Error during feature engineering: {e}")

    # Model Training Steps
    st.subheader("🏋️‍♂️ Model Training")

    st.write("""
    In this section, we will split the data into training and testing sets, train the RandomForestClassifier, 
    and evaluate its initial performance. 
    """)

    try:
        # 1. Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        st.write("### Data split into training and testing sets.")

        # 2. Training the RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        st.write("### Model training completed.")

        # 3. Predicting and evaluating on the test set
        y_pred = model.predict(X_test)

        st.write("### Initial Model Evaluation:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Save the trained model
        joblib.dump(model, "vehicle_transmission_model.pkl")
        st.success("Model trained and saved successfully.")

    except Exception as e:
        st.error(f"Error during model training: {e}")



# Model Evaluation Section
elif section == "Model Evaluation":
    st.title("🔍 Model Evaluation")
    try:
        # Load the model
        model = joblib.load("vehicle_transmission_model.pkl")
        st.success("Model loaded successfully.")

        le = LabelEncoder()

        # Prepare the data for evaluation
        X = merged_df[[
            "dealer_type", "stock_type", "mileage", "price", "model_year",
            "make", "model", "certified", "fuel_type_from_vin", "number_price_changes"
        ]].dropna()

        y = merged_df["transmission_from_vin"].loc[X.index]

        # Encode y_true using the same encoder
        y_encoded = le.fit_transform(y)

        # Encode categorical columns in X
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col].astype(str))

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict and evaluate
        y_pred = model.predict(X_scaled)

        # Specify the labels parameter to match the number of classes in y_encoded
        st.write("### Accuracy Score:", accuracy_score(y_encoded, y_pred))
        st.write("### Classification Report:")
        st.text(classification_report(y_encoded, y_pred, labels=[0, 1], target_names=['A', 'M']))
        st.write("### Confusion Matrix:")
        st.write(confusion_matrix(y_encoded, y_pred))
        
    except FileNotFoundError:
        st.warning("Model file not found. Please train the model first.")
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("""
    Explore and analyze the dataset using an interactive Power BI dashboard. 
    The dashboard provides insights and visualizations on transmission types, pricing trends, and more.
    """)

    st.write("The Power BI dashboard will be embedded here soon.") 

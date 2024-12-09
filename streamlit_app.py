import os
import streamlit as st
import pandas as pd
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
if section == "Dashboard":
    st.title("🚗 Vehicle Transmission Classifier")
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
    st.write(
        """ In the initial phase of this project, we performed Exploratory Data Analysis (EDA) to gain a deeper understanding of the dataset and its characteristics. This process included cleaning the data, addressing missing values, and resolving inconsistencies, such as handling outliers and data imbalance. We examined the distribution of numerical features such as vehicle year, price, and mileage, while also exploring relationships between categorical features like make, model, and dealer information.
            The EDA revealed key patterns and correlations in the data, such as newer vehicles and specific brands being more likely to have automatic transmissions. Visualizations, including bar charts and heatmaps, effectively highlighted these insights.
        """
    )
    st.subheader("Dataset Information")
    st.image("info1.jpeg", caption="Dataset Overview - Part 1")
    st.image("info2.jpeg", caption="Dataset Overview - Part 2")
    st.subheader("Visualizations")
    st.image("chart7.jpeg", caption="Transmission Distribution (Auto vs Manual)")
    st.image("chart2.png", caption="Price vs Mileage Scatter Plot")
    st.image("plt3.png", caption="Correlation Heatmap")

# Model Building Section
elif section == "Model Building":
    st.title("🧑‍🔬 Model Building")
    st.write(
        """
        The model was pre-trained and saved as `model.pkl`. 
        This section confirms that the model and other required files were successfully loaded.
        """
    )
    st.write(
        """ The model was built to evaluate the performance of various machine learning classifiers in predicting the target variable. The process began by selecting a diverse range of models, including Logistic Regression, K-Nearest Neighbors, Naive Bayes, Support Vector Machines, Random Forest, Decision Tree, and XGBoost. To handle missing values, a `SimpleImputer` was employed to replace them with a constant (0), ensuring consistency across all folds of the training data. Each model was incorporated into a pipeline alongside the imputer, streamlining the preprocessing and training stages.
        To ensure robust evaluation, 5-fold cross-validation was conducted for each pipeline. This method split the data into training and testing subsets in multiple iterations, calculating the accuracy for each fold. The mean and standard deviation of these accuracy scores provided insights into the performance and stability of the models. The results were stored systematically, allowing for easy comparison and enabling the selection of the most effective classifier for the dataset. This approach ensured fairness in evaluation and enhanced the reliability of the chosen model.
        """
    )

    try:
        # Load the saved model and preprocessing files
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        original_columns = joblib.load('models/original_columns.pkl')

        st.success("Model and preprocessing files loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model files: {e}")

# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")

    try:
        # Load model and preprocessing files
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        original_columns = joblib.load('models/original_columns.pkl')

        st.subheader("Enter Vehicle Details:")
        dealer_type = st.selectbox("Dealer Type", merged_df['dealer_type'].unique())
        stock_type = st.selectbox("Stock Type", merged_df['stock_type'].unique())
        mileage = st.number_input("Mileage", min_value=0)
        price = st.number_input("Price", min_value=0)
        model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
        make = st.selectbox("Make", merged_df['make'].unique())
        available_models = merged_df[merged_df['make'] == make]['model'].unique()
        model_input = st.selectbox("Model", available_models)
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

        if st.button("Generate Prediction"):
            try:
                # Reindex input data to match original columns
                input_df = input_data.reindex(columns=original_columns, fill_value=0)

                # Scale the input data
                scaled_input = scaler.transform(input_df)

                # Make a prediction
                prediction = model.predict(scaled_input)

                # Map prediction to human-readable label
                transmission_mapping = {0: "Automatic", 1: "Manual"}
                predicted_transmission = transmission_mapping.get(prediction[0], "Unknown")

                st.write(f"### Predicted Transmission: {predicted_transmission}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    except Exception as e:
        st.error(f"Error loading files: {e}")

# Power BI Dashboard Section
elif section == "Power BI Dashboard":
    st.title("📊 Power BI Dashboard")
    st.write("Click [here](https://app.powerbi.com/groups/me/reports/c9772dbc-0131-4e5a-a559-43a5c22874b3/ca237ccb0ae673ae960a?experience=power-bi) to view the Power BI dashboard.")


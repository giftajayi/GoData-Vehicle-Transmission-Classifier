import streamlit as st
import pandas as pd

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "Visualization"])

# Define paths to CSV files
csv_files = {
    "Main Dataset": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data1.csv",
    "Dataset 2": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data2.csv",
    "Dataset 3": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data3.csv",
    "Dataset 4": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data4.csv",
    "Dataset 5": "https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data5.csv",
}

# Load all datasets
dataframes = {}
for name, path in csv_files.items():
    try:
        dataframes[name] = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load {name}: {e}")

# Check column names in all datasets
st.write("### Columns in Each Dataset")
for name, df in dataframes.items():
    st.write(f"**{name}**")
    st.write(df.columns.tolist())

# Define the common key (update based on inspection of column names)
common_key = "vehicle_id"  # Replace with the correct column name if necessary

try:
    # Merge datasets iteratively
    merged_df = dataframes["Main Dataset"]
    for name, df in dataframes.items():
        if name != "Main Dataset":
            if common_key in df.columns:
                merged_df = merged_df.merge(df, on=common_key, how="inner")
            else:
                st.warning(f"Skipping merge with {name} as it lacks the common key '{common_key}'")

    st.success("Datasets successfully merged!")
    st.write(f"Merged Dataset Shape: {merged_df.shape}")
    st.dataframe(merged_df.head())

except Exception as e:
    st.error(f"An error occurred during the merge: {e}")

# Section: Introduction
if section == "Introduction":
    st.title("🚗 Vehicle Transmission Classifier")
    st.write("""
    This app demonstrates a machine learning workflow for classifying vehicle transmissions 
    (Automatic or Manual) based on various features like model year, make, mileage, price, and more.
    The dataset contains vehicle listings from Edmonton dealerships and additional related datasets.
    """)

    st.write("### Datasets Overview")
    for name, df in dataframes.items():
        st.subheader(name)
        st.write(f"**Shape:** {df.shape}")
        st.dataframe(df.head())

# Section: Visualization
elif section == "Visualization":
    st.title("📊 Data Visualizations")
    st.write("Explore key insights and trends in the dataset through the visualizations below:")

    # Visualizations 1 to 8
    visualizations = {
        "Transmission Type Distribution": "plt1.png",
        "Price vs Mileage Scatter Plot": "plt2.png",
        "Feature Correlation Heatmap": "plt3.png",
        "Model Year Distribution": "plt4.png",
        "Price Distribution by Fuel Type": "plt5.png",
        "Mileage Boxplot by Transmission Type": "plt6.png",
        "Price vs Model Year Trend": "plt7.png",
        "Make Popularity Countplot": "plt8.png",
    }

    for caption, img_path in visualizations.items():
        st.image(img_path, caption=caption, use_column_width=True)

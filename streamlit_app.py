import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import pandas as pd

# Title of the App
st.title("🚗 Vehicle Transmission Classifier")

df = pd.read_csv("https://raw.githubusercontent.com/giftajayi/GoData-Vehicle-Transmission-Classifier/refs/heads/master/Cleaned_data1.csv")
df.head(10) 

### 1. Introduction Section ###
with st.expander("🔍 Introduction"):
    st.write("""
    This app demonstrates a machine learning workflow for classifying vehicle transmissions (Automatic or Manual) based on various features like model year, make, mileage, price, and more.
    The dataset contains vehicle listings from Edmonton dealerships.
    """)
    
### 2. Data Visualization ###
with st.expander("📊 Visualization"):

    # Visualization 1: Transmission Type Distribution
    image_url_1 = "plt1.png"
    st.image(image_url_1, caption="Transmission Type Distribution", use_column_width=True)

    # Visualization 2: Price vs Mileage Scatter Plot
    image_url_2 = "plt2.png"
    st.image(image_url_2, caption="Price vs Mileage Scatter Plot", use_column_width=True)

    # Visualization 3: Correlation Heatmap
    image_url_3 = "plt3.png"
    st.image(image_url_3, caption="Feature Correlation Heatmap", use_column_width=True)

    # Visualization 4: Model Year Distribution
    image_url_4 = "plt 4.png"
    st.image(image_url_4, caption="Model Year Distribution", use_column_width=True)

    # Visualization 5: Price Distribution by Fuel Type
    image_url_5 = "plt5.png"
    st.image(image_url_5, caption="Price Distribution by Fuel Type", use_column_width=True)

    # Visualization 6: Mileage Boxplot by Transmission Type
    image_url_6 = "plt6.png"
    st.image(image_url_6, caption="Mileage Boxplot by Transmission Type", use_column_width=True)

    # Visualization 7: Price vs Model Year Line Plot
    image_url_7 = "plt7.png"
    st.image(image_url_7, caption="Price vs Model Year Trend", use_column_width=True)

    # Visualization 8: Make Popularity Countplot
    image_url_8 = "plt8.png"
    st.image(image_url_8, caption="Make Popularity Countplot", use_column_width=True)

    st.write("These visualizations provide insights into the dataset and model performance.")



   




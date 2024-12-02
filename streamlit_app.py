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

### 2. Dataset Overview and EDA ###
with st.expander("📊 Exploratory Data Analysis (EDA)"):
    image_url = "plt1.png"
    st.image(image_url, caption="Visualization from GitHub", use_column_width=True)

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
### 2. Dataset Overview and EDA ###
# if 'df' in locals():
    # with st.expander("📊 Exploratory Data Analysis (EDA)"):

### 2. Data Visualization###
with st.expander("📊 Visualization)"):
    image_url = "plt1.png"
    st.image(image_url, caption="Visualization from GitHub", use_column_width=True)
    
   



   




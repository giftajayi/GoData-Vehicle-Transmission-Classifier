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


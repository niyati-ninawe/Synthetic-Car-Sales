import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import io

# --- SETUP ---
st.set_page_config(page_title="Vehicle Subscription Dashboard", layout="wide")
st.title("🚗 Vehicle Subscription Analysis Platform")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_vehicle_subscription_dataset.csv")
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Navigation")
tab = st.sidebar.radio("Go to:", [
    "Data Visualization", 
    "Classification", 
    "Clustering", 
    "Association Rule Mining", 
    "Regression"])

# --- COMMON LABEL ENCODING ---
le_dict = {}
label_df = df.copy()
for col in label_df.select_dtypes(include='object'):
    le = LabelEncoder()
    label_df[col] = le.fit_transform(label_df[col].astype(str))
    le_dict[col] = le

# --- TAB 1: DATA VISUALIZATION ---
if tab == "Data Visualization":
    st.header("📊 Descriptive Insights")
    st.write("Below are 10+ insightful visualizations from the data:")

    fig1 = px.histogram(df, x="Monthly_Income", nbins=50, title="Monthly Income Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, y="Subscription_Budget", title="Subscription Budget (Outlier Detection)")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.pie(df, names="Willingness_Subscribe", title="Willingness to Subscribe")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(df, x="Age", color="Gender", barmode='overlay', title="Age vs Gender")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(df, x="Monthly_Income", y="Subscription_Budget", color="Occupation",
                      title="Budget vs Income by Occupation")
    st.plotly_chart(fig5, use_container_width=True)

    durations = df["Preferred_Duration"].value_counts(dropna=False).reset_index()
    durations.columns = ["Duration", "Count"]
    fig6 = px.bar(durations, x="Duration", y="Count", title="Preferred Subscription Duration")
    st.plotly_chart(fig6, use_container_width=True)

    fig7 = px.histogram(df, x="NPS_Score", nbins=20, title="NPS Score Distribution")
    st.plotly_chart(fig7, use_container_width=True)

    swap_val = df["Swap_Valuable"].value_counts(dropna=False).reset_index()
    swap_val.columns = ["Swap_Option", "Count"]
    fig8 = px.bar(swap_val, x="Swap_Option", y="Count", title="Is Swap Feature Valuable?")
    st.plotly_chart(fig8, use_container_width=True)

    st.write("Raw Data Preview")
    st.dataframe(df.head())
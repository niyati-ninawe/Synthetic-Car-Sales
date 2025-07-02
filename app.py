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

    fig6 = px.bar(df["Preferred_Duration"].value_counts().reset_index(), x="index", y="Preferred_Duration",
                  title="Preferred Subscription Duration")
    st.plotly_chart(fig6, use_container_width=True)

    fig7 = px.histogram(df, x="NPS_Score", nbins=20, title="NPS Score Distribution")
    st.plotly_chart(fig7, use_container_width=True)

    fig8 = px.bar(df["Swap_Valuable"].value_counts().reset_index(), x="index", y="Swap_Valuable",
                  title="Is Swap Feature Valuable?")
    st.plotly_chart(fig8, use_container_width=True)

    st.write("Raw Data Preview")
    st.dataframe(df.head())

# --- TAB 2: CLASSIFICATION ---
elif tab == "Classification":
    st.header("🤖 Classification: Willingness Prediction")
    st.write("Models: KNN, Decision Tree, Random Forest, Gradient Boosting")

    target = "Willingness_Subscribe"
    X = label_df.drop(target, axis=1)
    y = label_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append([
            name,
            round(accuracy_score(y_test, y_pred), 2),
            round(precision_score(y_test, y_pred, average='weighted'), 2),
            round(recall_score(y_test, y_pred, average='weighted'), 2),
            round(f1_score(y_test, y_pred, average='weighted'), 2)
        ])

    st.subheader("Model Comparison Table")
    st.table(pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]))

    selected_model = st.selectbox("Select model for Confusion Matrix:", list(models.keys()))
    model = models[selected_model]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write(f"Confusion Matrix: {selected_model}")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, prob[:, 1], pos_label=1)
            ax.plot(fpr, tpr, label=name)
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Upload New Data to Predict")
    upload_file = st.file_uploader("Upload CSV", type="csv")
    if upload_file:
        new_data = pd.read_csv(upload_file)
        new_data_encoded = new_data.copy()
        for col in new_data_encoded.columns:
            if col in le_dict:
                new_data_encoded[col] = le_dict[col].transform(new_data_encoded[col].astype(str))
        predictions = model.predict(new_data_encoded)
        new_data["Predicted_Willingness"] = le_dict[target].inverse_transform(predictions)
        st.dataframe(new_data)
        csv_out = new_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv_out, "predictions.csv")

# --- TAB 3: CLUSTERING ---
elif tab == "Clustering":
    st.header("🔍 Customer Segmentation via KMeans")
    st.subheader("Elbow Chart")
    distortions = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(label_df)
        distortions.append(kmeans.inertia_)
    fig = plt.figure()
    plt.plot(k_range, distortions, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    st.pyplot(fig)

    k_val = st.slider("Select Number of Clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k_val, random_state=42).fit(label_df)
    df_clustered = df.copy()
    df_clustered["Cluster"] = kmeans.labels_
    st.subheader("Clustered Data Sample")
    st.dataframe(df_clustered.head())

    cluster_csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", cluster_csv, "clustered_data.csv")

# --- TAB 4: ASSOCIATION RULE MINING ---
elif tab == "Association Rule Mining":
    st.header("🔗 Association Rule Mining")
    st.write("Based on 'Motivating_Features' and 'Subscription_Concerns'")
    df_apriori = df[["Motivating_Features", "Subscription_Concerns"]].copy()
    basket = df_apriori.apply(lambda row: list(set(str(row["Motivating_Features"]).split(", ") + str(row["Subscription_Concerns"]).split(", "))), axis=1)
    te = TransactionEncoder()
    te_ary = te.fit(basket).transform(basket)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.6)
    freq_items = apriori(df_tf, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    st.write("Top 10 Rules by Confidence")
    st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

# --- TAB 5: REGRESSION ---
elif tab == "Regression":
    st.header("📈 Regression Analysis")
    st.write("Target: Subscription_Budget")

    target = "Subscription_Budget"
    X = label_df.drop(target, axis=1)
    y = label_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressors = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    for name, model in regressors.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader(f"{name} Regressor")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted - {name}")
        st.pyplot(fig)
# Streamlit App Code Goes Here (To Be Generated)

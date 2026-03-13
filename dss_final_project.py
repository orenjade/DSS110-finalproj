import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Screentime, Sleep & Stress Analysis", layout="wide")

st.title("📱 Screentime, Sleep, and Stress Analysis")
st.markdown("Analyzing how smartphone usage habits influence **Work Productivity Score** using machine learning.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000.csv")
    df = df.sample(10000, random_state=42)
    df = df.drop_duplicates()
    return df

df = load_data()

# --- Sidebar ---
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Model Training & Results", "Predict Productivity"])

# =====================
# SECTION 1: Data Overview
# =====================
if section == "Data Overview":
    st.header("📊 Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str).rename("Data Type"))

# =====================
# SECTION 2: EDA
# =====================
elif section == "EDA":
    st.header("🔍 Exploratory Data Analysis")

    # Histogram
    st.subheader("Distribution of Daily Phone Hours")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Daily_Phone_Hours'], kde=True, ax=ax)
    ax.set_xlabel("Daily Phone Hours")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close()

    # Scatter plot with jitter
    st.subheader("Daily Phone Hours vs. Work Productivity Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    jitter = np.random.rand(len(df)) * 0.5 - 0.25
    sns.scatterplot(x=df['Daily_Phone_Hours'], y=df['Work_Productivity_Score'] + jitter, alpha=0.4, ax=ax)
    ax.set_xlabel("Daily Phone Hours")
    ax.set_ylabel("Work Productivity Score (with Jitter)")
    st.pyplot(fig)
    plt.close()

    # Line graph
    st.subheader("Average Work Productivity Score vs. Daily Phone Hours")
    avg_productivity = df.groupby('Daily_Phone_Hours')['Work_Productivity_Score'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Daily_Phone_Hours', y='Work_Productivity_Score', data=avg_productivity, ax=ax)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Daily Phone Hours")
    ax.set_ylabel("Average Work Productivity Score")
    st.pyplot(fig)
    plt.close()

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    st.pyplot(fig)
    plt.close()

# =====================
# SECTION 3: Model Training
# =====================
elif section == "Model Training & Results":
    st.header("🤖 Model Training & Evaluation")

    @st.cache_data
    def train_models(df):
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop('Work_Productivity_Score', axis=1)
        y = df_encoded['Work_Productivity_Score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)

        # Random Forest
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)

        # Decision Tree
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train_scaled, y_train)
        y_pred_dt = dt.predict(X_test_scaled)

        return (lr, rf, dt, scaler, X, X_train, X_test, y_train, y_test,
                y_pred_lr, y_pred_rf, y_pred_dt)

    with st.spinner("Training models... this may take a moment."):
        lr, rf, dt, scaler, X, X_train, X_test, y_train, y_test, y_pred_lr, y_pred_rf, y_pred_dt = train_models(df)

    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Decision Tree"],
        "MAE": [
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_rf),
            mean_absolute_error(y_test, y_pred_dt)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            np.sqrt(mean_squared_error(y_test, y_pred_dt))
        ],
        "R² Score": [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_rf),
            r2_score(y_test, y_pred_dt)
        ]
    })

    st.subheader("Model Comparison")
    st.dataframe(results.set_index("Model").style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R² Score": "{:.4f}"}))

    # Feature Importance
    st.subheader("Top 10 Feature Importances (Random Forest)")
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_10 = feature_importances.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_10.values, y=top_10.index, hue=top_10.index, legend=False, palette='viridis', ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 Feature Importances")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("📝 Conclusion")
    st.markdown("""
    - All models showed **low R² scores**, suggesting the features don't fully explain productivity variance.
    - **Weekend Screen Time** and **Daily Phone Hours** were the most influential features.
    - More sophisticated models or additional features may be needed for better predictions.
    """)

# =====================
# SECTION 4: Predict
# =====================
elif section == "Predict Productivity":
    st.header("🔮 Predict Work Productivity Score")
    st.markdown("Enter values below to get a predicted **Work Productivity Score**.")

    col1, col2 = st.columns(2)
    with col1:
        daily_phone = st.slider("Daily Phone Hours", 0.0, 16.0, 4.0)
        social_media = st.slider("Social Media Hours", 0.0, 10.0, 2.0)
        sleep_hours = st.slider("Sleep Hours", 3.0, 12.0, 7.0)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    with col2:
        weekend_screen = st.slider("Weekend Screen Time Hours", 0.0, 16.0, 5.0)
        app_usage = st.slider("App Usage Count", 1, 50, 10)
        caffeine = st.slider("Caffeine Intake (cups)", 0, 10, 2)
        age = st.number_input("Age", min_value=15, max_value=80, value=25)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    device = st.selectbox("Device Type", ["Android", "iOS"])

    if st.button("Predict", type="primary"):
        @st.cache_data
        def get_trained_rf(df):
            df_encoded = pd.get_dummies(df, drop_first=True)
            X = df_encoded.drop('Work_Productivity_Score', axis=1)
            y = df_encoded['Work_Productivity_Score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train_scaled, y_train)
            return rf, scaler, X.columns.tolist()

        rf_model, scaler_model, columns = get_trained_rf(df)

        new_data = {
            'Daily_Phone_Hours': daily_phone,
            'Social_Media_Hours': social_media,
            'Sleep_Hours': sleep_hours,
            'Stress_Level': stress_level,
            'Weekend_Screen_Time_Hours': weekend_screen,
            'App_Usage_Count': app_usage,
            'Caffeine_Intake_Cups': caffeine,
            'Age': age,
            'Gender': gender,
            'Device_Type': device
        }

        new_df = pd.DataFrame([new_data])
        new_encoded = pd.get_dummies(new_df, drop_first=True)
        for col in columns:
            if col not in new_encoded.columns:
                new_encoded[col] = 0
        new_encoded = new_encoded[columns]

        scaled = scaler_model.transform(new_encoded)
        prediction = rf_model.predict(scaled)[0]

        st.success(f"### Predicted Work Productivity Score: **{prediction:.2f}**")

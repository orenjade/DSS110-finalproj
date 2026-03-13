import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Screentime & Productivity Study",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #F7F5F0;
    color: #1a1a1a;
}
[data-testid="stSidebar"] {
    background-color: #1C2B2D !important;
}
[data-testid="stSidebar"] * { color: #D9CFC0 !important; }

.main-title {
    font-family: 'Lora', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #1C2B2D;
    letter-spacing: -0.5px;
    margin-bottom: 0.1rem;
}
.main-subtitle {
    font-size: 1.05rem;
    font-weight: 300;
    color: #5a5a5a;
    margin-bottom: 2rem;
    font-style: italic;
}
.section-header {
    font-family: 'Lora', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #1C2B2D;
    border-bottom: 2px solid #C8B89A;
    padding-bottom: 0.4rem;
    margin-top: 1.5rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E0D8CC;
    border-left: 4px solid #1C2B2D;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #1C2B2D;
}
.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-top: 0.2rem;
}
.finding-card {
    background: #FFFFFF;
    border: 1px solid #E0D8CC;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    height: 100%;
}
.finding-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
.finding-title {
    font-family: 'Lora', serif;
    font-size: 1rem;
    font-weight: 600;
    color: #1C2B2D;
    margin-bottom: 0.3rem;
}
.finding-text { font-size: 0.9rem; color: #555; line-height: 1.6; }
.callout {
    background: #EEF2F0;
    border-left: 4px solid #4A7C6F;
    border-radius: 0 4px 4px 0;
    padding: 1rem 1.4rem;
    margin: 1.2rem 0;
    font-size: 0.95rem;
    color: #2a2a2a;
    line-height: 1.7;
}
.soft-divider {
    border: none;
    border-top: 1px solid #DDD6C8;
    margin: 2rem 0;
}
.stButton > button {
    background-color: #1C2B2D !important;
    color: #F7F5F0 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover { background-color: #2E4547 !important; }
</style>
""", unsafe_allow_html=True)

# ── Chart theme ───────────────────────────────────────────────────────────────
PALETTE = ["#1C2B2D", "#4A7C6F", "#C8B89A", "#8B9E9B", "#6B5344"]
sns.set_theme(style="whitegrid")
mpl.rcParams.update({
    "figure.facecolor": "#FAFAF8", "axes.facecolor": "#FAFAF8",
    "axes.edgecolor": "#CCCCCC", "axes.labelcolor": "#333333",
    "xtick.color": "#555", "ytick.color": "#555",
    "grid.color": "#E8E8E8", "axes.spines.top": False, "axes.spines.right": False,
})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Navigation")
    section = st.radio("", [
        "Overview", "Exploratory Analysis", "Key Findings",
        "Model Results", "Predict Score"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<small style='color:#8A9A9C'>DSS 110 · Final Project<br>Smartphone Usage & Productivity</small>",
        unsafe_allow_html=True
    )

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Screentime, Sleep &amp; Stress Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">How smartphone usage habits shape work productivity — a data science study</div>', unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000 (2).csv")
    df = df.sample(10000, random_state=42)
    return df.drop_duplicates()

df = load_data()

# ── Pre-train RF model at startup (cached) ───────────────────────────────────
@st.cache_resource
def get_rf(_df):
    df_enc = pd.get_dummies(_df, drop_first=True)
    X = df_enc.drop('Work_Productivity_Score', axis=1)
    y = df_enc['Work_Productivity_Score']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(sc.fit_transform(X_train), y_train)
    return rf, sc, X.columns.tolist()

rf_model, scaler_model, feature_cols = get_rf(df)

# ═════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════
if section == "Overview":
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{df.shape[0]:,}", df.shape[1],
         int(df.isnull().sum().sum()),
         df.select_dtypes(include='number').shape[1]],
        ["Observations", "Variables", "Missing Values", "Numeric Features"]
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Sample Records</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

# ═════════════════════════════════════════════
# EDA
# ═════════════════════════════════════════════
elif section == "Exploratory Analysis":
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.markdown("**Distribution of Daily Phone Hours**")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df['Daily_Phone_Hours'], kde=True, color=PALETTE[0], edgecolor="white", linewidth=0.5, ax=ax)
    ax.set_xlabel("Daily Phone Hours", labelpad=8)
    ax.set_ylabel("Count", labelpad=8)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown("**Daily Phone Hours vs. Work Productivity Score**")
    fig, ax = plt.subplots(figsize=(9, 4))
    jitter = np.random.rand(len(df)) * 0.5 - 0.25
    ax.scatter(df['Daily_Phone_Hours'], df['Work_Productivity_Score'] + jitter,
               alpha=0.25, s=10, color=PALETTE[1])
    ax.set_xlabel("Daily Phone Hours", labelpad=8)
    ax.set_ylabel("Work Productivity Score", labelpad=8)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown("**Average Productivity Score by Daily Phone Hours**")
    avg = df.groupby('Daily_Phone_Hours')['Work_Productivity_Score'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(x='Daily_Phone_Hours', y='Work_Productivity_Score', data=avg,
                 color=PALETTE[0], linewidth=2, ax=ax)
    ax.fill_between(avg['Daily_Phone_Hours'], avg['Work_Productivity_Score'], alpha=0.08, color=PALETTE[0])
    ax.set_xlabel("Daily Phone Hours", labelpad=8)
    ax.set_ylabel("Avg. Productivity Score", labelpad=8)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown("**Correlation Matrix**")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                linewidths=0.5, linecolor="#F0EDE8", annot_kws={"size": 8}, ax=ax)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ═════════════════════════════════════════════
# KEY FINDINGS
# ═════════════════════════════════════════════
elif section == "Key Findings":
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout">
    Based on analysis of 10,000 sampled records, the following patterns were identified through
    exploratory analysis and machine learning modelling.
    </div>
    """, unsafe_allow_html=True)

    findings = [
        ("📱", "Screen Time is the Strongest Predictor",
         "Weekend screen time and daily phone hours ranked as the top two most important features in "
         "the Random Forest model, suggesting that overall device usage — not just work-hour usage — "
         "has a measurable relationship with productivity."),
        ("😴", "Sleep Hours Matter",
         "Sleep hours appeared among the top influential features. Users with lower sleep hours tended "
         "to show greater variability in productivity scores, reinforcing research linking adequate rest "
         "to cognitive performance."),
        ("📲", "Social Media Usage Shows Moderate Impact",
         "Social media hours ranked third in feature importance. Time spent on social platforms during "
         "the day correlates with reduced work output, though the effect is secondary to total screen time."),
        ("☕", "Caffeine &amp; Stress Have Smaller but Present Effects",
         "Stress level and caffeine intake showed moderate feature importance — secondary to screen time "
         "and sleep, but still contributing meaningfully to the model's predictions."),
        ("🤖", "Models Struggled to Predict Productivity",
         "All three models returned low or negative R² scores (−0.03 to −1.02), suggesting the available "
         "features do not fully explain productivity variance. Psychological and environmental factors "
         "likely play a larger, uncaptured role."),
        ("📊", "No Single Feature Dominates",
         "The correlation matrix showed no feature with a strong linear relationship to Work Productivity "
         "Score, supporting the view that productivity is a multifactorial outcome resistant to simple "
         "single-variable prediction."),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, text) in enumerate(findings):
        target = col1 if i % 2 == 0 else col2
        target.markdown(f"""
        <div class="finding-card">
            <div class="finding-icon">{icon}</div>
            <div class="finding-title">{title}</div>
            <div class="finding-text">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Implications &amp; Next Steps</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout">
    <strong>For future work:</strong> Incorporating psychological well-being metrics, job type, work environment,
    and notification frequency could substantially improve predictive power. Advanced models such as gradient
    boosting or neural networks, combined with richer feature sets, may better capture the complexity of
    productivity as a behavioral outcome.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# MODEL RESULTS
# ═════════════════════════════════════════════
elif section == "Model Results":
    st.markdown('<div class="section-header">Model Training &amp; Evaluation</div>', unsafe_allow_html=True)

    @st.cache_data
    def train_models(df):
        df_enc = pd.get_dummies(df, drop_first=True)
        X = df_enc.drop('Work_Productivity_Score', axis=1)
        y = df_enc['Work_Productivity_Score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_train)
        Xte = sc.transform(X_test)
        lr = LinearRegression().fit(Xtr, y_train)
        rf = RandomForestRegressor(random_state=42).fit(Xtr, y_train)
        dt = DecisionTreeRegressor(random_state=42).fit(Xtr, y_train)
        return lr, rf, dt, sc, X, X_test, y_test, lr.predict(Xte), rf.predict(Xte), dt.predict(Xte)

    with st.spinner("Training models…"):
        lr, rf, dt, sc, X, X_test, y_test, p_lr, p_rf, p_dt = train_models(df)

    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Decision Tree"],
        "MAE":  [mean_absolute_error(y_test, p) for p in [p_lr, p_rf, p_dt]],
        "RMSE": [np.sqrt(mean_squared_error(y_test, p)) for p in [p_lr, p_rf, p_dt]],
        "R²":   [r2_score(y_test, p) for p in [p_lr, p_rf, p_dt]],
    })

    st.markdown("**Performance Comparison**")
    st.dataframe(
        results.set_index("Model").style
            .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"})
            .background_gradient(subset=["R²"], cmap="RdYlGn"),
        use_container_width=True
    )

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown("**Top 10 Feature Importances — Random Forest**")
    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(fi.index[::-1], fi.values[::-1], color=PALETTE[0], height=0.6)
    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va='center', fontsize=8, color="#555")
    ax.set_xlabel("Importance Score", labelpad=8)
    ax.set_xlim(0, fi.values.max() * 1.2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ═════════════════════════════════════════════
# PREDICT
# ═════════════════════════════════════════════
elif section == "Predict Score":
    st.markdown('<div class="section-header">Predict Work Productivity Score</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout">
    Enter values below to generate a predicted Work Productivity Score using the trained Random Forest model.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        daily_phone    = st.slider("Daily Phone Hours", 0.0, 16.0, 4.0, 0.5)
        social_media   = st.slider("Social Media Hours", 0.0, 10.0, 2.0, 0.5)
        sleep_hours    = st.slider("Sleep Hours", 3.0, 12.0, 7.0, 0.5)
        stress_level   = st.slider("Stress Level (1–10)", 1, 10, 5)
    with col2:
        weekend_screen = st.slider("Weekend Screen Time Hours", 0.0, 16.0, 5.0, 0.5)
        app_usage      = st.slider("App Usage Count", 1, 50, 10)
        caffeine       = st.slider("Caffeine Intake (cups/day)", 0, 10, 2)
        age            = st.number_input("Age", min_value=15, max_value=80, value=25)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    device = st.selectbox("Device Type", ["Android", "iOS"])

    if st.button("Generate Prediction"):
        new_enc = pd.get_dummies(pd.DataFrame([{
            'Daily_Phone_Hours': daily_phone, 'Social_Media_Hours': social_media,
            'Sleep_Hours': sleep_hours, 'Stress_Level': stress_level,
            'Weekend_Screen_Time_Hours': weekend_screen, 'App_Usage_Count': app_usage,
            'Caffeine_Intake_Cups': caffeine, 'Age': age,
            'Gender': gender, 'Device_Type': device
        }]), drop_first=True)
        for c in feature_cols:
            if c not in new_enc.columns:
                new_enc[c] = 0
        pred = rf_model.predict(scaler_model.transform(new_enc[feature_cols]))[0]

        if pred <= 3.5:
            band_color = "#4A7C6F"
            band_label = "Low Phone Impact"
            band_desc  = "Productivity is largely unaffected by smartphone usage. Other factors like environment, workload, or sleep are likely the main drivers."
            band_range = "1.0 – 3.5"
        elif pred <= 5.5:
            band_color = "#8B7355"
            band_label = "Moderate Phone Impact"
            band_desc  = "Smartphone habits show a noticeable but moderate relationship with productivity. Reducing recreational screen time may yield meaningful gains."
            band_range = "3.6 – 5.5"
        elif pred <= 7.5:
            band_color = "#B05C3A"
            band_label = "High Phone Impact"
            band_desc  = "Smartphone usage patterns appear to significantly influence productivity. Weekend screen time and social media usage are likely key contributors."
            band_range = "5.6 – 7.5"
        else:
            band_color = "#8B2020"
            band_label = "Very High Phone Impact"
            band_desc  = "Productivity is strongly associated with heavy smartphone use. Structured digital habits or screen-time limits are strongly advised."
            band_range = "7.6 – 10.0"

        st.markdown(f"""
        <div style="background:#EEF2F0;border-left:4px solid #4A7C6F;border-radius:0 4px 4px 0;
                    padding:1.2rem 1.6rem;margin-top:1rem;">
            <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;color:#4A7C6F;">
                Predicted Score
            </div>
            <div style="font-family:'Lora',serif;font-size:2.8rem;font-weight:600;color:#1C2B2D;margin-top:0.2rem;">
                {pred:.2f}
            </div>
        </div>
        <div style="background:#FFFFFF;border:1px solid #E0D8CC;border-left:4px solid {band_color};
                    border-radius:0 4px 4px 0;padding:1.1rem 1.4rem;margin-top:0.8rem;">
            <div style="margin-bottom:0.4rem;">
                <span style="background:{band_color};color:#fff;font-size:0.75rem;font-weight:600;
                             text-transform:uppercase;letter-spacing:0.8px;padding:0.2rem 0.6rem;
                             border-radius:3px;">{band_label}</span>
                &nbsp;&nbsp;<span style="font-size:0.8rem;color:#888;">Score range: {band_range}</span>
            </div>
            <div style="font-size:0.92rem;color:#444;line-height:1.6;">{band_desc}</div>
        </div>
        <div style="margin-top:1.2rem;">
            <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#888;margin-bottom:0.6rem;">Score Scale</div>
            <div style="display:flex;flex-direction:column;gap:0.4rem;">
                <div style="display:flex;align-items:center;gap:0.7rem;">
                    <div style="width:12px;height:12px;border-radius:2px;background:#4A7C6F;flex-shrink:0;"></div>
                    <span style="font-size:0.85rem;color:#333;"><strong>1.0 – 3.5 &nbsp; Low Impact</strong> &nbsp;·&nbsp; Productivity largely unaffected by phone usage</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.7rem;">
                    <div style="width:12px;height:12px;border-radius:2px;background:#8B7355;flex-shrink:0;"></div>
                    <span style="font-size:0.85rem;color:#333;"><strong>3.6 – 5.5 &nbsp; Moderate Impact</strong> &nbsp;·&nbsp; Noticeable but manageable phone influence</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.7rem;">
                    <div style="width:12px;height:12px;border-radius:2px;background:#B05C3A;flex-shrink:0;"></div>
                    <span style="font-size:0.85rem;color:#333;"><strong>5.6 – 7.5 &nbsp; High Impact</strong> &nbsp;·&nbsp; Phone habits significantly shape productivity</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.7rem;">
                    <div style="width:12px;height:12px;border-radius:2px;background:#8B2020;flex-shrink:0;"></div>
                    <span style="font-size:0.85rem;color:#333;"><strong>7.6 – 10.0 &nbsp; Very High Impact</strong> &nbsp;·&nbsp; Heavy usage strongly associated with lower output</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

"""
Simple Anchor Orbit web UI — no terminal needed after this.

Run with:
    streamlit run app.py

Two tabs:
1. Single Customer — fill a form, get an instant prediction
2. Batch Upload — upload a CSV of many customers, get a ranked table
"""
import importlib

st = importlib.import_module("streamlit")
import pandas as pd
import joblib
import yaml

from src.predict import clean_new_data, engineer_new_features, align_to_training_columns

st.set_page_config(page_title="Anchor Orbit", page_icon="◉", layout="centered")

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main, .main, .block-container { background: linear-gradient(135deg, #9edfd0 0%, #a9cbed 100%) !important; }
    .stApp { background: linear-gradient(135deg, #9edfd0 0%, #a9cbed 100%) !important; }
    [data-testid="stHeader"] { background: #12304a !important; }
    .stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"], .stApp [data-baseweb="select"], .stApp input { color: #111827 !important; }
    .ao-header { background: linear-gradient(110deg, #12304a, #176b76); border-left: 6px solid #20a486; padding: 14px 18px; margin: 0 0 18px; color: #ffffff; }
    .ao-kicker { color: #a9f0d9; font-size: .72rem; font-weight: 700; letter-spacing: .08em; }
    .ao-header h1 { color: #ffffff !important; margin: 2px 0 0; }
    .ao-header p { color: #d9f7ee !important; margin: 3px 0 0; }
    h1 { color: #12304a; letter-spacing: 0; margin-bottom: 0; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #c6e3dc; padding: 12px; transition: transform .2s, border-color .2s; }
    [data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #20a486; }
    .stButton > button { background: #12304a; color: #ffffff; border: 0; transition: background .2s, transform .2s; }
    .stButton > button:hover { background: #20a486; transform: translateY(-1px); }
    [data-baseweb="tab"] { transition: color .2s; }
    [data-baseweb="tab"]:hover { color: #20a486; }
    [data-baseweb="select"] > div, input, textarea { border-color: #b9d8d2 !important; }
    [data-baseweb="select"] > div:focus-within, input:focus, textarea:focus { border-color: #20a486 !important; box-shadow: 0 0 0 1px #20a486 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_everything():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    bundle = joblib.load(cfg["model"]["output_path"])
    return cfg, bundle


cfg, bundle = load_everything()
model, scaler, feature_names = bundle["model"], bundle["scaler"], bundle["feature_names"]
cols = cfg["data"]["columns"]

st.markdown('<div class="ao-header"><div class="ao-kicker">ANCHOR ORBIT / RETENTION INTELLIGENCE</div><h1>◉ Anchor Orbit</h1><p>Turn churn signals into timely customer action.</p></div>', unsafe_allow_html=True)
st.subheader("Customer Churn Predictor")
st.caption("Retention signals, ranked by business impact.")
st.caption(f"Model: {bundle['model_name']}  ·  Test F1: {bundle['metrics']['f1']:.3f}  ·  "
           f"ROC-AUC: {bundle['metrics']['roc_auc']:.3f}")

tab1, tab2 = st.tabs(["🔍 Single Customer", "📂 Batch Upload"])

# ------------------------------------------------------------------
# TAB 1 — single customer form
# ------------------------------------------------------------------
with tab1:
    st.subheader("Review customer signals")

    col_a, col_b = st.columns(2)
    with col_a:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col_b:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)))

    if st.button("Run Risk Check", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "customerID": "MANUAL-ENTRY",
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])

        cleaned = clean_new_data(row)
        feat_df = engineer_new_features(cleaned, cfg)
        X = align_to_training_columns(feat_df, feature_names)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        priority_score = prob * monthly_charges

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.1%}")
        c2.metric("Prediction", "⚠️ Will Churn" if prob >= 0.5 else "✅ Will Stay")
        c3.metric("Retention Priority Score", f"{priority_score:.1f}")

        st.progress(min(prob, 1.0))
        if prob >= 0.7:
            st.error("High risk — prioritize this customer for retention outreach.")
        elif prob >= 0.4:
            st.warning("Moderate risk — worth monitoring.")
        else:
            st.success("Low risk — likely to stay.")

# ------------------------------------------------------------------
# TAB 2 — batch CSV upload
# ------------------------------------------------------------------
with tab2:
    st.subheader("Upload customer data")
    st.caption("Same columns as the original dataset, without the Churn column.")

    uploaded = st.file_uploader("Choose CSV file", type="csv")
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        cleaned = clean_new_data(raw)
        feat_df = engineer_new_features(cleaned, cfg)
        X = align_to_training_columns(feat_df, feature_names)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]

        result = pd.DataFrame({
            cols["customer_id"]: feat_df[cols["customer_id"]],
            "churn_probability": probs.round(3),
            "will_likely_churn": probs >= 0.5,
            "monthly_revenue": feat_df[cols["revenue"]],
            "retention_priority_score": (probs * feat_df[cols["revenue"]]).round(2),
        }).sort_values("retention_priority_score", ascending=False)

        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download predictions as CSV",
            result.to_csv(index=False),
            "predictions.csv",
            "text/csv",
        )

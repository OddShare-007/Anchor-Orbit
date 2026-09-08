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
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main, .main {
        background: #eef5f3 !important;
        color: #183243 !important;
    }

    .block-container { max-width: 980px; padding: 2.5rem 2rem 4rem; }
    [data-testid="stHeader"] { background: #12304a !important; }
    [data-testid="stToolbar"] button, [data-testid="stToolbar"] button * { color: #ffffff !important; }
    [data-testid="stToolbar"] button svg { fill: #ffffff !important; stroke: #ffffff !important; color: #ffffff !important; }
    [data-testid="stToolbar"] button svg path[fill="none"] { fill: none !important; }
    [data-testid="stToolbar"] button:hover { background: rgba(255, 255, 255, .14) !important; }

    .stApp,
    .stApp p,
    .stApp label,
    .stApp div,
    .stApp span,
    .stApp li,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-baseweb="select"],
    .stApp [data-baseweb="popover"],
    .stApp [role="listbox"],
    .stApp [role="option"],
    .stApp input,
    .stApp textarea,
    .stApp [data-testid="stSelectbox"] *,
    .stApp [data-baseweb="popover"] *,
    .stApp [role="listbox"] *,
    .stApp [role="option"] * {
        color: #183243 !important;
    }

    [data-testid="stHeader"] *,
    [data-testid="stToolbar"] *,
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] button *,
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] button *,
    header button,
    header button *,
    .ao-header,
    .ao-header *,
    .ao-kicker,
    .ao-header h1,
    .ao-header p {
        color: #ffffff !important;
    }

    [data-testid="stHeader"] button svg,
    [data-testid="stHeader"] button svg *,
    [data-testid="stToolbar"] button svg,
    [data-testid="stToolbar"] button svg *,
    header button svg,
    header button svg * {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div *,
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [role="listbox"],
    [role="listbox"] *,
    [role="option"],
    [role="option"] * {
        color: #183243 !important;
        background: #ffffff !important;
    }

    [data-baseweb="select"] svg,
    [data-baseweb="popover"] svg,
    [data-testid="stSelectbox"] svg,
    [data-baseweb="select"] svg * {
        fill: #476270 !important;
        stroke: #476270 !important;
        color: #476270 !important;
    }

    [role="option"][aria-selected="true"],
    [role="option"][aria-selected="true"] *,
    [role="option"][aria-selected="true"] > div {
        color: #12304a !important;
        background: #dff3ed !important;
    }

    [role="option"]:hover,
    [role="option"]:hover *,
    [role="option"]:hover > div {
        color: #12304a !important;
        background: #eef8f5 !important;
    }

    .ao-header {
        background: linear-gradient(110deg, #12304a, #176b76);
        border-left: 6px solid #20a486;
        padding: 1.35rem 1.5rem;
        margin: 0 0 1.5rem;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(18, 48, 74, .12);
    }
    .ao-kicker, .ao-kicker * {
        color: #ffffff !important;
        font-size: .72rem !important;
        font-weight: 700 !important;
        letter-spacing: .08em !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    .ao-header h1 { color: #ffffff !important; margin: 2px 0 0; }
    .ao-header p { color: #d9f7ee !important; margin: 3px 0 0; }
    h1, h2, h3 { color: #12304a !important; letter-spacing: 0; }
    [data-testid="stCaptionContainer"] { color: #476270 !important; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #c6e3dc; padding: 1rem; transition: transform .2s, border-color .2s; }
    [data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #20a486; }
    .stButton > button,
    .stButton > button *,
    .stButton > button span,
    .stButton > button p {
        color: #ffffff !important;
    }
    .stButton > button { background: #12304a; border: 0; min-height: 2.75rem; font-weight: 700; transition: background .2s, transform .2s; }
    .stButton > button:hover { background: #20a486; transform: translateY(-1px); }
    [data-baseweb="tab"] { transition: color .2s; }
    [data-baseweb="tab"]:hover { color: #20a486; }
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #12304a, #176b76) !important;
        border-color: rgba(255, 255, 255, .45) !important;
    }
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] label,
    [data-testid="stFileUploaderDropzone"] strong {
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] svg,
    [data-testid="stFileUploaderDropzone"] svg *,
    [data-testid="stFileUploaderDropzone"] path,
    [data-testid="stFileUploaderDropzone"] path * {
        fill: #000000 !important;
        stroke: #000000 !important;
        color: #000000 !important;
    }
    [data-baseweb="select"] > div, input, textarea { border-color: #b9d8d2 !important; background: #ffffff !important; }
    [data-baseweb="input"] input, [data-baseweb="input"] input::placeholder { color: #183243 !important; }
    [data-baseweb="select"] > div:focus-within, input:focus, textarea:focus { border-color: #20a486 !important; box-shadow: 0 0 0 1px #20a486 !important; }
    .ao-section-label { color: #176b76; font-size: .76rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; border-bottom: 1px solid #c6e3dc; padding-bottom: .45rem; margin: .75rem 0 .8rem; }
    .ao-result { background: #ffffff; border: 1px solid #c6e3dc; padding: 1rem 1.15rem; margin: 1rem 0; }
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
st.caption("Assess one customer in seconds, or rank an entire outreach list by revenue at risk.")
st.caption(f"Model: {bundle['model_name']}  ·  Test F1: {bundle['metrics']['f1']:.3f}  ·  "
           f"ROC-AUC: {bundle['metrics']['roc_auc']:.3f}")

tab1, tab2 = st.tabs(["🔍 Single Customer", "📂 Batch Upload"])

# ------------------------------------------------------------------
# TAB 1 — single customer form
# ------------------------------------------------------------------
with tab1:
    st.subheader("Review customer signals")
    st.caption("Use the latest account details. The score combines churn likelihood with monthly revenue.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="ao-section-label">Customer profile</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="ao-section-label">Plan and billing</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="ao-section-label">Decision</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="ao-result">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# TAB 2 — batch CSV upload
# ------------------------------------------------------------------
with tab2:
    st.subheader("Upload customer data")
    st.caption("Upload the same fields as the training data, without the Churn column. Results are sorted by revenue at risk.")

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

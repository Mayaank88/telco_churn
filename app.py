"""
Telco Customer Churn Analytics & Prediction System
Data Warehouse & Data Mining (DWM) - Problem-Based Learning (PBL) project

Run:
    pip install -r requirements.txt
    streamlit run app.py

Optional (auto-detected, graceful fallbacks if absent):
    xgboost        -> XGBoost model (else sklearn GradientBoostingClassifier)
    imbalanced-learn -> SMOTE option (else class weights / none)
"""
import os
import re
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    raise SystemExit("Streamlit is not installed. Run: pip install streamlit")

# --------------------------------------------------------------------------
# Dependency check (hard requirements) + optional libraries (fallbacks)
# --------------------------------------------------------------------------
_MISSING = []
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    _MISSING.append("plotly")
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score,
                                 roc_curve)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
except ImportError:
    _MISSING.append("scikit-learn")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

st.set_page_config(page_title="Telco Churn Analytics (DWM PBL)", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

if _MISSING:
    st.error("Missing required packages: **%s**\n\nInstall with:\n\n`pip install %s`"
             % (", ".join(_MISSING), " ".join(_MISSING)))
    st.stop()

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
CSV_PATH = "telecom_churn_10k.csv"
N_RECORDS = 10_500
DEFAULT_SEED = 42
RAW_COLS = ["CustomerID", "Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure",
            "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Churn"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"]
ADOPTION_COLS = ["PhoneService", "MultipleLines"] + ADDON_COLS
SERVICE_COLS_ALL = ["PhoneService", "MultipleLines", "InternetService"] + ADDON_COLS
CONTRACT_COLS = ["Contract", "PaperlessBilling", "PaymentMethod"]
BINARY_COLS = ["Partner", "Dependents", "PaperlessBilling", "PhoneService"]
OHE_COLS = ["MultipleLines", "InternetService"] + ADDON_COLS + ["Contract", "PaymentMethod"]
NUM_COLS = ["Tenure", "MonthlyCharges", "TotalCharges"]
BUCKETS = ["0-12", "13-24", "25-48", "49+"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)"]
RISK_LOW, RISK_HIGH = 0.35, 0.65
RISK_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

st.markdown("""
<style>
.kpi{background:rgba(128,128,128,.10);padding:14px 18px;border-radius:12px;margin-bottom:8px}
.kpi-l{font-size:.85rem;opacity:.75;text-transform:uppercase;letter-spacing:.04em}
.kpi-v{font-size:1.9rem;font-weight:700;line-height:1.2}
.kpi-s{font-size:.8rem;opacity:.65}
.stage{background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.5);
       border-radius:12px;padding:14px 16px;min-height:150px}
.stage h4{margin:0 0 6px 0}
.arrow{font-size:2.2rem;text-align:center;padding-top:50px}
.badge{display:inline-block;padding:6px 16px;border-radius:20px;color:white;font-weight:700}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small UI helpers (version-tolerant across Streamlit releases)
# --------------------------------------------------------------------------
def pchart(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)


def dframe(obj, **kw):
    try:
        st.dataframe(obj, width="stretch", **kw)
    except Exception:
        st.dataframe(obj, use_container_width=True, **kw)


def kpi(label, value, sub="", color="#6366f1"):
    return ("<div class='kpi' style='border-left:6px solid %s'><div class='kpi-l'>%s</div>"
            "<div class='kpi-v'>%s</div><div class='kpi-s'>%s</div></div>"
            % (color, label, value, sub))


def snake(name):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name).lower()


# --------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATOR
# --------------------------------------------------------------------------
def generate_dataset(n=N_RECORDS, seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    suffix = ["".join(r) for r in rng.choice(letters, size=(n, 5))]
    ids = ["%05d-%s" % (i + 1, s) for i, s in enumerate(suffix)]

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.binomial(1, 0.16, n)
    partner = np.where(rng.random(n) < 0.48, "Yes", "No")
    dependents = np.where(rng.random(n) < np.where(partner == "Yes", 0.40, 0.12), "Yes", "No")

    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.21, 0.24])
    t_m2m = np.clip(np.round(rng.gamma(1.4, 14, n)), 1, 72)
    t_1y = rng.integers(6, 73, n)
    t_2y = rng.integers(12, 73, n)
    tenure = np.where(contract == "Month-to-month", t_m2m,
                      np.where(contract == "One year", t_1y, t_2y)).astype(int)

    paperless = np.where(rng.random(n) < np.where(contract == "Month-to-month", 0.70, 0.45),
                         "Yes", "No")
    payment = rng.choice(PAYMENTS, n, p=[0.34, 0.19, 0.22, 0.25])

    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    phone = np.where(rng.random(n) < 0.90, "Yes", "No")
    phone[(phone == "No") & (internet == "No")] = "Yes"  # at least one service
    multi = np.where(phone == "No", "No phone service",
                     np.where(rng.random(n) < 0.42, "Yes", "No"))

    def addon(p_dsl, p_fiber):
        p = np.where(internet == "Fiber optic", p_fiber, p_dsl)
        yes = rng.random(n) < p
        return np.where(internet == "No", "No internet service", np.where(yes, "Yes", "No"))

    sec, bkp = addon(0.37, 0.22), addon(0.35, 0.35)
    dev, tech = addon(0.34, 0.36), addon(0.37, 0.22)
    tv, mov = addon(0.30, 0.50), addon(0.30, 0.52)

    base = (np.where(phone == "Yes", 20.0, 0.0) + np.where(multi == "Yes", 5.0, 0.0)
            + np.where(internet == "DSL", 25.0, np.where(internet == "Fiber optic", 50.0, 0.0)))
    for col in (sec, bkp, dev, tech):
        base += np.where(col == "Yes", 5.0, 0.0)
    for col in (tv, mov):
        base += np.where(col == "Yes", 8.0, 0.0)
    monthly = np.round(np.clip(base + rng.normal(0, 3, n), 18.25, 120), 2)
    total = np.round(tenure * monthly * rng.uniform(0.96, 1.04, n), 2)

    # Churn logit with the requested correlations
    fiber = internet == "Fiber optic"
    z = (1.35 * (contract == "Month-to-month") - 0.45 * (contract == "One year")
         - 1.20 * (contract == "Two year")
         + 0.55 * fiber + 0.90 * (fiber & (tech != "Yes"))
         + 0.35 * ((internet != "No") & (sec == "No"))
         + 0.035 * (monthly - 65)
         + 1.00 * (tenure < 12) + 0.40 * ((tenure >= 12) & (tenure < 24)) - 0.015 * tenure
         + 0.55 * (payment == "Electronic check") + 0.25 * senior
         + 0.15 * (paperless == "Yes") - 0.25 * (partner == "Yes")
         - 0.30 * (dependents == "Yes") - 0.50 * (internet == "No")
         + rng.normal(0, 0.35, n))
    lo, hi = -8.0, 4.0  # calibrate intercept -> ~27% overall churn
    for _ in range(40):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > 0.27:
            hi = mid
        else:
            lo = mid
    prob = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    df = pd.DataFrame({
        "CustomerID": ids, "Gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "Tenure": tenure, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total, "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": sec,
        "OnlineBackup": bkp, "DeviceProtection": dev, "TechSupport": tech,
        "StreamingTV": tv, "StreamingMovies": mov, "Churn": churn})
    # inject a few missing values so the ETL imputation step is meaningful
    df.loc[rng.choice(n, 25, replace=False), "TotalCharges"] = np.nan
    return df[RAW_COLS]


@st.cache_data(show_spinner="Loading dataset...")
def load_raw():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            if len(df) > 0 and set(RAW_COLS) <= set(df.columns):
                return df
        except Exception:
            pass
    df = generate_dataset()
    df.to_csv(CSV_PATH, index=False)
    return df


# --------------------------------------------------------------------------
# 2. ETL: CLEAN -> ENCODE -> SCALE -> LOAD (STAR SCHEMA)
# --------------------------------------------------------------------------
def clean_data(raw):
    df = raw.copy().drop_duplicates("CustomerID")
    for c in ["TotalCharges", "MonthlyCharges", "Tenure"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())
    df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median()).astype(int)
    # domain-based imputation: TotalCharges ~ Tenure x MonthlyCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(df["Tenure"] * df["MonthlyCharges"])
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    for c in [c for c in df.columns if df[c].dtype == object]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)
    df["TenureBucket"] = pd.cut(df["Tenure"], [-1, 12, 24, 48, 10 ** 6], labels=BUCKETS).astype(str)
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Running ETL...")
def get_clean(_raw, sig):
    return clean_data(_raw)


def encode_features(df):
    """Label-encode binaries, one-hot encode multi-class categoricals."""
    X = pd.DataFrame(index=df.index)
    X["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    X["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    for c in BINARY_COLS:
        X[c] = (df[c] == "Yes").astype(int)
    for c in NUM_COLS:
        X[c] = df[c].astype(float)
    ohe = pd.get_dummies(df[OHE_COLS], prefix=OHE_COLS, prefix_sep="=", dtype=int)
    return pd.concat([X, ohe], axis=1)


def make_scaler(name):
    return StandardScaler() if name == "StandardScaler" else MinMaxScaler()


def _build_warehouse(c):
    dim_customer = c[["CustomerID", "Gender", "SeniorCitizen", "Partner", "Dependents"]].copy()
    dim_customer.insert(0, "customer_key", np.arange(1, len(dim_customer) + 1))
    dim_services = c[SERVICE_COLS_ALL].drop_duplicates().reset_index(drop=True)
    dim_services.insert(0, "services_key", np.arange(1, len(dim_services) + 1))
    dim_contract = c[CONTRACT_COLS].drop_duplicates().reset_index(drop=True)
    dim_contract.insert(0, "contract_key", np.arange(1, len(dim_contract) + 1))
    months = np.arange(0, int(c["Tenure"].max()) + 1)
    dim_time = pd.DataFrame({
        "time_key": months + 1, "Tenure_Months": months,
        "Tenure_Quarter": np.where(months > 0, (months - 1) // 3 + 1, 0),
        "Tenure_Year": np.where(months > 0, (months - 1) // 12 + 1, 0),
        "Tenure_Bucket": pd.cut(months, [-1, 12, 24, 48, 10 ** 6], labels=BUCKETS).astype(str)})
    f = c.merge(dim_customer[["CustomerID", "customer_key"]], on="CustomerID", how="left")
    f = f.merge(dim_services, on=SERVICE_COLS_ALL, how="left")
    f = f.merge(dim_contract, on=CONTRACT_COLS, how="left")
    f["time_key"] = f["Tenure"] + 1
    fact = f[["customer_key", "services_key", "contract_key", "time_key",
              "MonthlyCharges", "TotalCharges", "ChurnFlag"]].copy()
    tables = {"Fact_Subscriptions_Churn": fact, "Dim_Customer": dim_customer,
              "Dim_Services": dim_services, "Dim_Contract_Billing": dim_contract,
              "Dim_Time": dim_time}
    for k, t in tables.items():
        t.columns = [snake(x) for x in t.columns]
    return tables


@st.cache_data(show_spinner="Building data warehouse...")
def build_warehouse(_clean, sig):
    return _build_warehouse(_clean)


# --------------------------------------------------------------------------
# 3. MACHINE LEARNING
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models (Logistic Regression, Random Forest, Boosting)...")
def train_models(_clean, sig, scaler_name, imbalance):
    X = encode_features(_clean)
    y = _clean["ChurnFlag"].values
    feature_cols = list(X.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = make_scaler(scaler_name)
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    X_tr_s[NUM_COLS] = scaler.fit_transform(X_tr[NUM_COLS])
    X_te_s[NUM_COLS] = scaler.transform(X_te[NUM_COLS])

    Xr, yr, cw = X_tr_s, y_tr, None
    if imbalance == "SMOTE" and HAS_SMOTE:
        Xr, yr = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)
    elif imbalance == "Class weights":
        cw = "balanced"
    pos_w = float((yr == 0).sum() / max((yr == 1).sum(), 1)) if cw else 1.0

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight=cw),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12,
                                                min_samples_leaf=5, class_weight=cw,
                                                n_jobs=-1, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08,
                                          subsample=0.9, colsample_bytree=0.9,
                                          eval_metric="logloss", scale_pos_weight=pos_w,
                                          tree_method="hist", n_jobs=-1, random_state=42)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9, random_state=42)

    rows, roc, imps, cms = [], {}, {}, {}
    for name, m in models.items():
        if name == "Gradient Boosting" and cw:
            m.fit(Xr, yr, sample_weight=compute_sample_weight("balanced", yr))
        else:
            m.fit(Xr, yr)
        proba = m.predict_proba(X_te_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred, zero_division=0),
                     "Recall": recall_score(y_te, pred),
                     "F1-Score": f1_score(y_te, pred),
                     "ROC-AUC": roc_auc_score(y_te, proba)})
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc[name] = (fpr, tpr)
        cms[name] = confusion_matrix(y_te, pred)
        raw_imp = np.abs(m.coef_[0]) if name == "Logistic Regression" else m.feature_importances_
        imps[name] = pd.Series(raw_imp, index=feature_cols).sort_values(ascending=False)
    metrics = pd.DataFrame(rows).set_index("Model")
    return {"models": models, "metrics": metrics, "roc": roc, "imps": imps, "cms": cms,
            "feature_cols": feature_cols, "scaler": scaler, "best": metrics["ROC-AUC"].idxmax(),
            "n_train": len(Xr), "n_test": len(X_te), "imbalance": imbalance}


@st.cache_data(show_spinner="Scoring customers for churn risk...")
def risk_scores(_clean, sig, scaler_name):
    """Out-of-fold churn probabilities (no in-sample leakage) used for 'revenue at stake'."""
    X = encode_features(_clean)
    X[NUM_COLS] = make_scaler(scaler_name).fit_transform(X[NUM_COLS])
    gb = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.1,
                                    subsample=0.9, random_state=42)
    p = cross_val_predict(gb, X, _clean["ChurnFlag"].values,
                          cv=StratifiedKFold(3, shuffle=True, random_state=42),
                          method="predict_proba")[:, 1]
    return pd.Series(p, name="RiskScore")


def prepare(df_raw_like, art):
    X = encode_features(df_raw_like).reindex(columns=art["feature_cols"], fill_value=0).copy()
    X[NUM_COLS] = art["scaler"].transform(X[NUM_COLS])
    return X


def risk_category(p):
    return "Low" if p < RISK_LOW else ("Medium" if p < RISK_HIGH else "High")


# --------------------------------------------------------------------------
# Load + ETL + sidebar
# --------------------------------------------------------------------------
raw = load_raw()
sig = "%d-%d-%d" % (len(raw), int(raw["Churn"].eq("Yes").sum()),
                    int(pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0).sum()))
clean = get_clean(raw, sig)

st.sidebar.title("📡 Telco Churn DWM")
st.sidebar.caption("OLAP filters apply to the Dashboard tab")
with st.sidebar.expander("⚙️ ML configuration", expanded=False):
    scaler_name = st.selectbox("Feature scaling", ["StandardScaler", "MinMaxScaler"])
    imb_opts = ["Class weights", "None"] + (["SMOTE"] if HAS_SMOTE else [])
    imbalance = st.selectbox("Imbalance handling", imb_opts)
    if not HAS_SMOTE:
        st.caption("Install `imbalanced-learn` to enable SMOTE.")
    st.caption("Boosting model: **%s**" % ("XGBoost" if HAS_XGB else "GradientBoosting (xgboost not found)"))
    thr = st.slider("High-risk score threshold", 0.30, 0.80, 0.50, 0.05)

st.sidebar.subheader("🧊 OLAP slice & dice")
f_int = st.sidebar.multiselect("Internet Service", sorted(clean["InternetService"].unique()),
                               default=sorted(clean["InternetService"].unique()))
f_pay = st.sidebar.multiselect("Payment Method", sorted(clean["PaymentMethod"].unique()),
                               default=sorted(clean["PaymentMethod"].unique()))
f_con = st.sidebar.multiselect("Contract Type", CONTRACTS, default=CONTRACTS)

with st.sidebar.expander("🔁 Regenerate dataset"):
    seed = st.number_input("Random seed", 0, 10 ** 6, DEFAULT_SEED)
    if st.button("Regenerate telecom_churn_10k.csv"):
        generate_dataset(seed=int(seed)).to_csv(CSV_PATH, index=False)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

art = train_models(clean, sig, scaler_name, imbalance)
scores = risk_scores(clean, sig, scaler_name)
scored = clean.join(scores)
warehouse = build_warehouse(clean, sig)

st.title("📡 Telco Customer Churn Analytics & Prediction System")
st.caption("Data Warehouse & Data Mining · Problem-Based Learning project · "
           "%s synthetic customers" % format(len(clean), ","))

tab_data, tab_etl, tab_dash, tab_ml, tab_doc = st.tabs(
    ["📁 Data Explorer", "🏗️ ETL & Schema", "📊 OLAP Dashboard", "🤖 ML & Predictor", "📘 PBL Documentation"])

# ==========================================================================
# TAB 1 - DATA EXPLORER
# ==========================================================================
with tab_data:
    st.subheader("Synthetic dataset (OLTP source)")
    st.caption("Auto-generated on first run and saved as `%s`." % CSV_PATH)
    c1, c2 = st.columns([1, 2])
    q = c1.text_input("🔎 Search Customer ID", placeholder="e.g. 00042")
    prow = c2.columns(2)
    t_rng = prow[0].slider("Tenure (months)", int(raw.Tenure.min()), int(raw.Tenure.max()),
                           (int(raw.Tenure.min()), int(raw.Tenure.max())))
    m_rng = prow[1].slider("Monthly charges ($)", float(raw.MonthlyCharges.min()),
                           float(raw.MonthlyCharges.max()),
                           (float(raw.MonthlyCharges.min()), float(raw.MonthlyCharges.max())))
    view = raw.copy()
    with st.expander("Filter by attributes", expanded=False):
        cols = st.columns(3)
        for i, col in enumerate(["Gender", "SeniorCitizen", "Partner", "Dependents", "Contract",
                                 "PaperlessBilling", "PaymentMethod", "InternetService", "Churn"]):
            opts = sorted(raw[col].dropna().unique().tolist())
            sel = cols[i % 3].multiselect(col, opts, default=opts, key="ex_" + col)
            view = view[view[col].isin(sel)]
    view = view[view.Tenure.between(*t_rng) & view.MonthlyCharges.between(*m_rng)]
    if q:
        view = view[view.CustomerID.str.contains(q.strip(), case=False, na=False)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows shown", format(len(view), ","))
    m2.metric("Churn rate (view)", "%.1f%%" % (100 * (view.Churn == "Yes").mean()) if len(view) else "-")
    m3.metric("Missing TotalCharges", int(raw.TotalCharges.isna().sum()))
    n_show = st.selectbox("Preview rows", [50, 100, 500, 1000, "All"], index=1)
    dframe(view if n_show == "All" else view.head(int(n_show)), height=380)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇️ Download telecom_churn_10k.csv", raw.to_csv(index=False).encode(),
                       "telecom_churn_10k.csv", "text/csv")
    d2.download_button("⬇️ Download filtered view", view.to_csv(index=False).encode(),
                       "telecom_churn_filtered.csv", "text/csv")
    with st.expander("Descriptive statistics"):
        dframe(raw.describe(include="all").T.astype(str))

# ==========================================================================
# TAB 2 - ETL & SCHEMA
# ==========================================================================
with tab_etl:
    st.subheader("ETL pipeline: OLTP source → Staging → Data Warehouse")
    Xenc = encode_features(clean)
    miss_before, miss_after = int(raw.isna().sum().sum()), int(clean.isna().sum().sum())
    fact = warehouse["Fact_Subscriptions_Churn"]
    a, b, c_, d, e = st.columns([5, 1, 5, 1, 5])
    a.markdown("<div class='stage'><h4>1 · EXTRACT</h4>Raw CSV / OLTP ingestion<br>"
               "• %s rows × %d columns<br>• %d missing values<br>• Source: <code>%s</code></div>"
               % (format(len(raw), ","), raw.shape[1], miss_before, CSV_PATH), unsafe_allow_html=True)
    b.markdown("<div class='arrow'>➜</div>", unsafe_allow_html=True)
    c_.markdown("<div class='stage'><h4>2 · TRANSFORM</h4>Cleaning &amp; feature engineering<br>"
                "• Type coercion, de-duplication, imputation<br>• Label + One-Hot encoding → %d features<br>"
                "• %s scaling on numeric columns</div>" % (Xenc.shape[1], scaler_name), unsafe_allow_html=True)
    d.markdown("<div class='arrow'>➜</div>", unsafe_allow_html=True)
    e.markdown("<div class='stage'><h4>3 · LOAD</h4>Staging → Data Warehouse<br>"
               "• Staging: %s rows<br>• 4 dimension tables + 1 fact table<br>• Fact rows: %s</div>"
               % (format(len(clean), ","), format(len(fact), ",")), unsafe_allow_html=True)

    if st.button("▶ Run ETL pipeline (live timings)"):
        with st.status("Running ETL...", expanded=True) as status:
            t = time.perf_counter()
            r_df = pd.read_csv(CSV_PATH)
            st.write("✅ **Extract** - %s rows read in %.3fs" % (format(len(r_df), ","), time.perf_counter() - t))
            t = time.perf_counter()
            c_df = clean_data(r_df)
            st.write("✅ **Transform (clean)** - %d → %d missing values in %.3fs"
                     % (int(r_df.isna().sum().sum()), int(c_df.isna().sum().sum()), time.perf_counter() - t))
            t = time.perf_counter()
            xe = encode_features(c_df)
            xe[NUM_COLS] = make_scaler(scaler_name).fit_transform(xe[NUM_COLS])
            st.write("✅ **Transform (encode + scale)** - %d features in %.3fs" % (xe.shape[1], time.perf_counter() - t))
            t = time.perf_counter()
            wh = _build_warehouse(c_df)
            st.write("✅ **Load** - %s tables, %s fact rows in %.3fs"
                     % (len(wh), format(len(wh["Fact_Subscriptions_Churn"]), ","), time.perf_counter() - t))
            status.update(label="ETL complete", state="complete")

    s1, s2, s3 = st.columns(3)
    with s1.expander("🧹 Cleaning & imputation", expanded=True):
        st.write("Missing values: **%d → %d**" % (miss_before, miss_after))
        st.caption("`TotalCharges` imputed as Tenure × MonthlyCharges; categoricals → mode; numerics → median.")
        dframe(raw.isna().sum().rename("missing (raw)").to_frame().query("`missing (raw)` > 0"))
    with s2.expander("🔢 Encoding", expanded=True):
        st.write("Label-encoded: %s" % ", ".join(BINARY_COLS + ["Gender"]))
        st.write("One-hot encoded: %d columns → %d dummy features" % (len(OHE_COLS), Xenc.shape[1] - 2 - len(BINARY_COLS) - len(NUM_COLS)))
        dframe(Xenc.head(5).T, height=200)
    with s3.expander("📏 Scaling (%s)" % scaler_name, expanded=True):
        scaled = pd.DataFrame(make_scaler(scaler_name).fit_transform(clean[NUM_COLS]), columns=NUM_COLS)
        cmp_ = pd.concat([clean[NUM_COLS].describe().T[["mean", "std", "min", "max"]].add_prefix("raw_"),
                          scaled.describe().T[["mean", "std", "min", "max"]].add_prefix("scaled_")], axis=1)
        dframe(cmp_.round(3))

    st.markdown("---")
    st.subheader("Dimensional model: Star schema")
    shapes, annos = [], []
    pos = {"Fact_Subscriptions_Churn": (0.5, 0.5, "#fde68a"), "Dim_Customer": (0.14, 0.80, "#bfdbfe"),
           "Dim_Services": (0.86, 0.80, "#bbf7d0"), "Dim_Contract_Billing": (0.14, 0.20, "#fbcfe8"),
           "Dim_Time": (0.86, 0.20, "#ddd6fe")}
    for name, (x, y, col) in pos.items():
        cols_ = list(warehouse[name].columns)
        hh = (len(cols_) + 1.6) * 0.0125
        hw = 0.13 if name != "Fact_Subscriptions_Churn" else 0.14
        if name != "Fact_Subscriptions_Churn":
            shapes.append(dict(type="line", x0=0.5, y0=0.5, x1=x, y1=y, line=dict(color="#6b7280", width=2),
                               layer="below"))
        shapes.append(dict(type="rect", x0=x - hw, x1=x + hw, y0=y - hh, y1=y + hh, fillcolor=col,
                           line=dict(color="#374151", width=2), layer="above"))
        lines = ["<b>%s</b>" % name]
        for i, cn in enumerate(cols_):
            tag = ("FK " if name.startswith("Fact") and cn.endswith("_key") else
                   "PK " if (not name.startswith("Fact") and i == 0) else "")
            lines.append("%s%s" % (tag, cn))
        annos.append(dict(x=x, y=y, text="<br>".join(lines), showarrow=False, align="left",
                          font=dict(size=11, color="#111827")))
    fig = go.Figure()
    fig.update_layout(shapes=shapes, annotations=annos, height=680, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]),
                      plot_bgcolor="rgba(0,0,0,0)")
    pchart(fig, key="star")
    t_sel = st.selectbox("Inspect warehouse table", list(warehouse.keys()))
    tcol1, tcol2 = st.columns([1, 3])
    tcol1.metric("Rows", format(len(warehouse[t_sel]), ","))
    tcol1.metric("Columns", warehouse[t_sel].shape[1])
    with tcol2:
        dframe(warehouse[t_sel].head(15), height=300)
    ok = (fact.customer_key.isin(warehouse["Dim_Customer"].customer_key).all()
          and fact.services_key.isin(warehouse["Dim_Services"].services_key).all()
          and fact.contract_key.isin(warehouse["Dim_Contract_Billing"].contract_key).all()
          and fact.time_key.isin(warehouse["Dim_Time"].time_key).all())
    if ok:
        st.success("Referential integrity: all fact foreign keys resolve to dimensions ✅")
    else:
        st.error("Referential integrity violation ❌")
    with st.expander("Example star-join query (OLAP over the warehouse)"):
        st.code("SELECT c.contract, c.payment_method, COUNT(*) AS customers,\n"
                "       AVG(f.churn_flag) AS churn_rate\n"
                "FROM Fact_Subscriptions_Churn f\n"
                "JOIN Dim_Contract_Billing c ON f.contract_key = c.contract_key\n"
                "GROUP BY c.contract, c.payment_method;", language="sql")
        j = fact.merge(warehouse["Dim_Contract_Billing"], on="contract_key")
        res = j.groupby(["contract", "payment_method"]).agg(
            customers=("churn_flag", "size"), churn_rate=("churn_flag", "mean")).reset_index()
        res["churn_rate"] = (100 * res["churn_rate"]).round(2)
        dframe(res)

    st.markdown("---")
    st.subheader("Feature correlation matrix")
    corr_full = Xenc.assign(Churn=clean["ChurnFlag"]).corr()
    topn = st.slider("Top-N features most correlated with Churn", 8, len(corr_full) - 1,
                     min(20, len(corr_full) - 1))
    sel = corr_full["Churn"].drop("Churn").abs().sort_values(ascending=False).head(topn).index.tolist() + ["Churn"]
    cfig = px.imshow(corr_full.loc[sel, sel], color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     aspect="auto", height=720)
    cfig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    pchart(cfig, key="corr")

# ==========================================================================
# TAB 3 - OLAP DASHBOARD
# ==========================================================================
with tab_dash:
    df = scored[scored.InternetService.isin(f_int) & scored.PaymentMethod.isin(f_pay)
                & scored.Contract.isin(f_con)]
    if df.empty:
        st.warning("No customers match the current slice. Adjust the sidebar filters.")
    else:
        active_risk = df[(df.Churn == "No") & (df.RiskScore >= thr)]
        at_stake = active_risk.MonthlyCharges.sum() * 12
        k = st.columns(4)
        k[0].markdown(kpi("Total Revenue", "${:,.0f}".format(df.TotalCharges.sum()),
                          "%s customers in slice" % format(len(df), ",")), unsafe_allow_html=True)
        k[1].markdown(kpi("Overall Churn Rate", "%.1f%%" % (100 * df.ChurnFlag.mean()),
                          "%s churned" % format(int(df.ChurnFlag.sum()), ","), "#ef4444"), unsafe_allow_html=True)
        k[2].markdown(kpi("High-Risk Revenue at Stake", "${:,.0f}".format(at_stake),
                          "annualised · %s active customers ≥ %d%% risk" % (format(len(active_risk), ","), int(thr * 100)),
                          "#f59e0b"), unsafe_allow_html=True)
        k[3].markdown(kpi("Avg Tenure", "%.1f mo" % df.Tenure.mean(), "avg monthly bill $%.2f" % df.MonthlyCharges.mean(),
                          "#22c55e"), unsafe_allow_html=True)

        st.markdown("#### Churn by tenure & contract type")
        meas = st.radio("Measure", ["Churn rate (%)", "Churned customers"], horizontal=True, key="m1")
        g = df.groupby(["TenureBucket", "Contract"]).agg(customers=("ChurnFlag", "size"),
                                                         churned=("ChurnFlag", "sum")).reset_index()
        g["rate"] = 100 * g.churned / g.customers
        rate_mode = meas.startswith("Churn rate")
        fig1 = px.bar(g, x="TenureBucket", y="rate" if rate_mode else "churned", color="Contract",
                      barmode="group" if rate_mode else "stack", text_auto=".1f" if rate_mode else True,
                      category_orders={"TenureBucket": BUCKETS, "Contract": CONTRACTS},
                      labels={"rate": "Churn rate (%)", "churned": "Churned customers",
                              "TenureBucket": "Tenure bucket (months)"},
                      hover_data=["customers"])
        pchart(fig1, key="bar1")

        st.markdown("#### Monthly vs total charges vs churn")
        samp = df.sample(min(3500, len(df)), random_state=1)
        fig2 = px.scatter(samp, x="MonthlyCharges", y="TotalCharges", color="Churn", opacity=0.6,
                          color_discrete_map={"Yes": "#ef4444", "No": "#3b82f6"},
                          hover_data=["CustomerID", "Tenure", "Contract", "InternetService"])
        pchart(fig2, key="scatter")

        st.markdown("#### Service adoption impact on churn")
        view_t = st.radio("Chart type", ["Heatmap", "Grouped bars"], horizontal=True, key="m3")
        recs = []
        for c in ADOPTION_COLS:
            for v, gp in df.groupby(c):
                recs.append({"Service": c, "Value": v, "Churn rate (%)": 100 * gp.ChurnFlag.mean(),
                             "Customers": len(gp)})
        sv = pd.DataFrame(recs)
        if view_t == "Heatmap":
            piv = sv.pivot(index="Service", columns="Value", values="Churn rate (%)")
            fig3 = px.imshow(piv, text_auto=".1f", color_continuous_scale="RdYlGn_r", aspect="auto", height=380)
        else:
            fig3 = px.bar(sv, x="Service", y="Churn rate (%)", color="Value", barmode="group",
                          hover_data=["Customers"])
        pchart(fig3, key="svc")

        st.markdown("#### OLAP pivot (dice) & roll-up")
        dims = ["Contract", "InternetService", "PaymentMethod", "TenureBucket", "Gender", "SeniorCitizen",
                "PaperlessBilling", "TechSupport", "OnlineSecurity", "Partner", "Dependents"]
        o1, o2, o3 = st.columns(3)
        r_dim = o1.selectbox("Rows", dims, index=0)
        c_dim = o2.selectbox("Columns", dims, index=3)
        o_meas = o3.selectbox("Measure", ["Churn rate (%)", "Customers", "Total revenue ($)",
                                          "Avg monthly charge ($)"])
        if r_dim == c_dim:
            st.info("Choose two different dimensions to build a pivot.")
        else:
            d = df.copy()
            d["SeniorCitizen"] = d["SeniorCitizen"].map({0: "No", 1: "Yes"})
            spec = {"Churn rate (%)": ("ChurnFlag", "mean", ".1f"), "Customers": ("ChurnFlag", "count", ",.0f"),
                    "Total revenue ($)": ("TotalCharges", "sum", ",.0f"),
                    "Avg monthly charge ($)": ("MonthlyCharges", "mean", ".1f")}
            vcol, agg, fmt = spec[o_meas]
            pv = d.pivot_table(index=r_dim, columns=c_dim, values=vcol, aggfunc=agg)
            if o_meas.startswith("Churn"):
                pv = pv * 100
            fig4 = px.imshow(pv, text_auto=fmt, color_continuous_scale="Viridis", aspect="auto",
                             labels=dict(color=o_meas))
            pchart(fig4, key="pivot")
        ru1, ru2 = st.columns([1, 1])
        level = ru1.radio("Roll-up level (tenure hierarchy)", ["Month", "Quarter", "Year"], horizontal=True)
        split = ru2.checkbox("Split by contract type", value=True)
        d = df.copy()
        d["Level"] = {"Month": d.Tenure, "Quarter": (d.Tenure - 1) // 3 + 1, "Year": (d.Tenure - 1) // 12 + 1}[level]
        keys = ["Level"] + (["Contract"] if split else [])
        ru = d.groupby(keys).agg(rate=("ChurnFlag", "mean"), customers=("ChurnFlag", "size")).reset_index()
        ru["rate"] *= 100
        fig5 = px.line(ru, x="Level", y="rate", color="Contract" if split else None, markers=True,
                       hover_data=["customers"], labels={"rate": "Churn rate (%)", "Level": "Tenure " + level})
        pchart(fig5, key="rollup")

# ==========================================================================
# TAB 4 - ML & PREDICTOR
# ==========================================================================
with tab_ml:
    st.subheader("Model evaluation")
    st.caption("Hold-out test set: %s customers · training rows after balancing: %s · scaler: %s · imbalance: %s"
               % (format(art["n_test"], ","), format(art["n_train"], ","), scaler_name, art["imbalance"]))
    mt = art["metrics"]
    try:
        dframe(mt.style.format("{:.3f}").highlight_max(axis=0, color="#14532d"))
    except Exception:
        dframe(mt.round(3))
    st.success("Best model by ROC-AUC: **%s** (%.3f)" % (art["best"], mt.loc[art["best"], "ROC-AUC"]))

    v1, v2 = st.columns(2)
    rfig = go.Figure()
    for name, (fpr, tpr) in art["roc"].items():
        rfig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="%s (AUC %.3f)" % (name, mt.loc[name, "ROC-AUC"])))
    rfig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Chance"))
    rfig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate",
                       height=420, legend=dict(x=0.45, y=0.05))
    with v1:
        pchart(rfig, key="roc")
    with v2:
        m_sel = st.selectbox("Model for importance / confusion matrix", list(art["models"]), 
                             index=list(art["models"]).index(art["best"]))
        imp = art["imps"][m_sel].head(15).iloc[::-1]
        ttl = "Top-15 |coefficients|" if m_sel == "Logistic Regression" else "Top-15 feature importances"
        pchart(px.bar(imp, x=imp.values, y=imp.index, orientation="h", title=ttl,
                      labels={"x": "importance", "y": ""}, height=380), key="imp")
    cm = art["cms"][m_sel]
    cfig2 = px.imshow(cm, text_auto=True, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                      color_continuous_scale="Blues", title="Confusion matrix - %s" % m_sel, height=330)
    pchart(cfig2, key="cm")

    st.markdown("---")
    st.subheader("🔮 Real-time churn predictor")

    FEATS = ["Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure", "Contract", "PaperlessBilling",
             "PaymentMethod", "MonthlyCharges", "PhoneService", "MultipleLines", "InternetService"] + ADDON_COLS
    base_profile = {c: (float(clean[c].median()) if c == "MonthlyCharges" else
                        int(clean[c].median()) if c in ("Tenure", "SeniorCitizen") else
                        clean[c].mode().iloc[0]) for c in FEATS}

    def normalize(p):
        p = dict(p)
        for c in ADDON_COLS:
            if p["InternetService"] == "No":
                p[c] = "No internet service"
            elif p[c] == "No internet service":
                p[c] = "No"
        if p["PhoneService"] == "No":
            p["MultipleLines"] = "No phone service"
        elif p["MultipleLines"] == "No phone service":
            p["MultipleLines"] = "No"
        return p

    def to_frame(rows):
        d = pd.DataFrame([normalize(r) for r in rows])
        d["TotalCharges"] = d["Tenure"] * d["MonthlyCharges"]
        return d

    _fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)

    @_fragment
    def predictor_panel():
        pm = st.selectbox("Prediction model", list(art["models"]), index=list(art["models"]).index(art["best"]),
                          key="pred_model")
        c1, c2, c3 = st.columns(3)
        P = {}
        P["Gender"] = c1.selectbox("Gender", ["Male", "Female"])
        P["SeniorCitizen"] = 1 if c1.selectbox("Senior citizen", ["No", "Yes"]) == "Yes" else 0
        P["Partner"] = c1.selectbox("Partner", ["No", "Yes"])
        P["Dependents"] = c1.selectbox("Dependents", ["No", "Yes"])
        P["Tenure"] = c1.slider("Tenure (months)", 1, 72, 6)
        P["Contract"] = c2.selectbox("Contract", CONTRACTS)
        P["PaperlessBilling"] = c2.selectbox("Paperless billing", ["Yes", "No"])
        P["PaymentMethod"] = c2.selectbox("Payment method", PAYMENTS)
        P["MonthlyCharges"] = c2.slider("Monthly charges ($)", 18.0, 120.0, 85.0, 0.5)
        P["PhoneService"] = c2.selectbox("Phone service", ["Yes", "No"])
        P["InternetService"] = c2.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        if P["PhoneService"] == "No":
            P["MultipleLines"] = c2.selectbox("Multiple lines", ["No phone service"], disabled=True)
        else:
            P["MultipleLines"] = c2.selectbox("Multiple lines", ["No", "Yes"])
        for col in ADDON_COLS:
            if P["InternetService"] == "No":
                P[col] = c3.selectbox(col, ["No internet service"], disabled=True, key="pd_" + col)
            else:
                P[col] = c3.selectbox(col, ["No", "Yes"], key="pd_" + col)

        P = normalize(P)
        variants = [P] + [dict(P, **{f: base_profile[f]}) for f in FEATS]
        probs = art["models"][pm].predict_proba(prepare(to_frame(variants), art))[:, 1]
        p, pv = float(probs[0]), probs[1:]
        cat = risk_category(p)

        g1, g2 = st.columns([1, 1])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=100 * p, number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Churn probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[cat]},
                   "steps": [{"range": [0, RISK_LOW * 100], "color": "rgba(34,197,94,.25)"},
                             {"range": [RISK_LOW * 100, RISK_HIGH * 100], "color": "rgba(245,158,11,.25)"},
                             {"range": [RISK_HIGH * 100, 100], "color": "rgba(239,68,68,.25)"}]}))
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
        with g1:
            pchart(gauge, key="gauge")
            st.markdown("Risk category: <span class='badge' style='background:%s'>%s RISK</span>"
                        % (RISK_COLORS[cat], cat.upper()), unsafe_allow_html=True)
            st.caption("Low < %d%% ≤ Medium < %d%% ≤ High" % (RISK_LOW * 100, RISK_HIGH * 100))
        # driver attribution: change in probability vs a "typical customer" value of each feature
        delta = pd.Series(p - pv, index=FEATS)
        lab = {f: "%s = %s" % (f, P[f]) for f in FEATS}
        top = pd.concat([delta.sort_values(ascending=False).head(6), delta.sort_values().head(3)])
        top = top[~top.index.duplicated()].sort_values()
        top = top[top.abs() > 0.002]
        with g2:
            if top.empty:
                st.info("This profile is close to the typical customer; no single dominant factor.")
            else:
                dfig = px.bar(x=100 * top.values, y=[lab[i] for i in top.index], orientation="h",
                              color=np.where(top.values > 0, "Raises risk", "Lowers risk"),
                              color_discrete_map={"Raises risk": "#ef4444", "Lowers risk": "#22c55e"},
                              labels={"x": "Δ churn probability vs typical value (pp)", "y": ""},
                              title="Top contributing factors", height=340)
                dfig.update_layout(legend_title_text="")
                pchart(dfig, key="drivers")
        tips = []
        if P["Contract"] == "Month-to-month":
            tips.append("Offer a discounted 1- or 2-year contract upgrade.")
        if P["InternetService"] == "Fiber optic" and P["TechSupport"] != "Yes":
            tips.append("Bundle free/discounted Tech Support for fiber customers.")
        if P["PaymentMethod"] == "Electronic check":
            tips.append("Incentivise auto-pay (bank transfer / credit card).")
        if P["Tenure"] < 12:
            tips.append("Enrol in the first-year onboarding & loyalty programme.")
        if P["MonthlyCharges"] > 85:
            tips.append("Review plan fit - high bill is a known churn driver.")
        if tips and cat != "Low":
            st.markdown("**Suggested retention actions**\n\n" + "\n".join("- " + t for t in tips))

    predictor_panel()

# ==========================================================================
# TAB 5 - PBL DOCUMENTATION
# ==========================================================================
with tab_doc:
    st.markdown("""
## 1. Problem statement & objectives
**Problem.** Customer churn is the largest recurring revenue leak for telecom operators; acquiring a new subscriber
costs several times more than retaining an existing one. Operational (OLTP) data is scattered across billing, CRM and
service systems, so analysts cannot easily see *who* is likely to leave, *why*, and *how much revenue is at risk*.

**Objectives**
1. Build a consolidated analytical store (data warehouse, star schema) from raw subscriber data.
2. Perform OLAP analysis (slice, dice, roll-up, drill-down) to locate high-churn segments.
3. Train and compare data-mining classifiers that predict churn probability for each customer.
4. Deliver a real-time scoring tool and actionable retention strategies.

## 2. DWM core concepts applied
| Concept | Where it appears in this app |
|---|---|
| **Data quality** | De-duplication, type coercion, missing-value imputation (`TotalCharges`), integrity checks |
| **ETL** | Extract CSV → Transform (clean, encode, scale) → Load into staging and warehouse tables |
| **Dimensional modeling** | Star schema: `Fact_Subscriptions_Churn` + `Dim_Customer`, `Dim_Services`, `Dim_Contract_Billing`, `Dim_Time` |
| **OLAP slice** | Fix one dimension (e.g. Contract = Month-to-month) via sidebar filters |
| **OLAP dice** | Select sub-cubes across several dimensions (Internet × Payment × Contract) |
| **Roll-up / drill-down** | Tenure hierarchy Month → Quarter → Year; pivot with swappable dimensions |
| **Measures** | Revenue, churn rate, customer count, average tenure, revenue at stake |

## 3. Data-mining techniques & ML methodology
- **Data:** synthetic 10,500-customer dataset with injected causal structure: month-to-month contracts, fiber users
  without tech support, high monthly charges, tenure < 12 months and electronic-check payers churn more.
- **Preprocessing:** label encoding (binary), one-hot encoding (multi-class), Standard/MinMax scaling of numerics.
- **Imbalance:** class weights (default), SMOTE (if `imbalanced-learn` is installed) or none; applied to training data only.
- **Models:** Logistic Regression (interpretable baseline), Random Forest (bagged trees, non-linear interactions),
  XGBoost or Gradient Boosting (sequential boosting).
- **Evaluation:** stratified 80/20 hold-out; Accuracy, Precision, Recall, F1, ROC-AUC, ROC curves, confusion matrix.
  Recall and F1 matter most for the churn class because missed churners are costlier than false alarms.
- **Explainability:** global feature importances; per-customer factor attribution by swapping each feature to its
  typical value and measuring the change in predicted probability.
- **Revenue at stake:** annualised monthly charges of active customers whose out-of-fold risk score exceeds the threshold.

## 4. Strategic business recommendations
1. **Convert month-to-month customers to annual plans** using discounts or device subsidies, especially in months 1-12.
2. **Bundle Tech Support & Online Security** with fiber plans; customers without them churn markedly more.
3. **Run a first-year onboarding programme** (welcome call, usage check at months 3 and 6) to cut early-tenure churn.
4. **Promote auto-pay** over electronic-check/manual payment methods with small bill credits.
5. **Review fiber pricing / value** for customers with high monthly charges and few add-ons.
6. **Operationalise the model:** score the base weekly, route High-risk customers to retention agents, and measure
   uplift with A/B tests before scaling campaigns.

## 5. Limitations & future work
The dataset is synthetic, so absolute numbers illustrate methodology rather than a real operator. Next steps:
real CDR/CRM data, SHAP explanations, hyper-parameter tuning, survival analysis (time-to-churn), and a
slowly-changing-dimension (SCD Type 2) design for customer history.
""")

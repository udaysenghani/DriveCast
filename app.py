# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(layout="wide", page_title="KPI Forecaster & What-If Simulator")
st.title("🚗 DriveCast-Car Dealership KPI Forecaster & What-If Simulator")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def ensure_monthly_continuity(df):
    # expects df with 'account_id' and 'date' columns
    full_frames = []
    for acc_id, g in df.groupby("account_id", group_keys=False):
        g = g.sort_values("date").copy()
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
        g2 = g.set_index("date").reindex(idx)
        g2.index.name = "date"
        # forward/backfill metadata (account_id, english_name, dealer_code)
        g2["account_id"] = acc_id
        if "english_name" in g.columns:
            g2["english_name"] = g["english_name"].ffill().bfill().iloc[0]
        if "dealer_code" in g.columns:
            g2["dealer_code"] = g["dealer_code"].ffill().bfill().iloc[0]
        g2["year"] = g2.index.year
        g2["month"] = g2.index.month
        full_frames.append(g2.reset_index())
    out = pd.concat(full_frames, ignore_index=True)
    # reorder cols
    return out

def recompute_ytd(df):
    tmp = df.copy()
    tmp["mv_nonan"] = tmp["monthly_value"].fillna(0)
    tmp["yearly_value_calc"] = tmp.groupby(["account_id","year"])["mv_nonan"].cumsum()
    df["yearly_value"] = tmp["yearly_value_calc"]
    return df.drop(columns=["yearly_value_calc","mv_nonan"], errors="ignore")

def small_sarimax_grid_search(y, seasonal_periods=12, max_p=2, max_q=2, max_P=1, max_Q=1):
    """
    Small grid search for SARIMAX hyperparams by AIC.
    Returns (order, seasonal_order) or (None, None) if not enough data.
    """
    y = y.dropna()
    if len(y) < 18:
        return None, None
    best_aic = np.inf
    best_order = None
    best_seas = None
    pdq = [(p, d, q) for p in range(0, max_p+1) for d in (0,1) for q in range(0, max_q+1)]
    PDQ = [(P, D, Q, seasonal_periods) for P in range(0, max_P+1) for D in (0,1) for Q in range(0, max_Q+1)]
    for order in pdq:
        for seas in PDQ:
            try:
                mod = SARIMAX(y, order=order, seasonal_order=seas,
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = order
                    best_seas = seas
            except Exception:
                continue
    return best_order, best_seas

def seasonal_naive_forecast(y, steps=3, m=12):
    y = y.dropna()
    if len(y) < m:
        return pd.Series([y.iloc[-1]]*steps, index=pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS"))
    last_season = y.iloc[-m:]
    reps = int(np.ceil(steps/m))
    vals = np.tile(last_season.values, reps)[:steps]
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    return pd.Series(vals, index=idx)

def forecast_kpi(y, steps=3):
    """
    Return forecast Series, ci (DataFrame with lower and upper), and model info.
    """
    y = y.dropna()
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    order, seas = small_sarimax_grid_search(y)
    if order is None:
        fc = seasonal_naive_forecast(y, steps=steps)
        return fc, None, ("seasonal_naive", None, None)
    try:
        mod = SARIMAX(y, order=order, seasonal_order=seas,
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        pred = res.get_forecast(steps=steps)
        fc = pd.Series(pred.predicted_mean, index=idx)
        ci = pred.conf_int(alpha=0.05)
        ci.index = idx
        return fc, ci, ("sarimax", order, seas)
    except Exception as e:
        # fallback
        fc = seasonal_naive_forecast(y, steps=steps)
        return fc, None, ("seasonal_naive", None, None)

@st.cache_data
def build_pivot(df):
    pivot = df.pivot_table(index="date", columns="account_id", values="monthly_value")
    return pivot

def compute_elasticities_loglog(wide_df, target_id, driver_ids):
    """
    Fit log-log OLS: log(target) ~ log(drivers). Return dict of elasticities.
    Requires positive values (we add small eps).
    """
    eps = 1e-6
    sub = wide_df[[target_id] + driver_ids].dropna()
    if sub.shape[0] < max(12, len(driver_ids)*3):
        return {}  # not enough data
    X = np.log(sub[driver_ids].replace(0, eps))
    y = np.log(sub[target_id].replace(0, eps))
    # add const
    X_const = sm.add_constant(X)
    try:
        model = sm.OLS(y, X_const).fit()
        coefs = model.params.drop("const", errors="ignore").to_dict()
        return coefs
    except Exception:
        return {}

def to_excel_bytes(df):
    buf = BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf

# -------------------------
# UI: Upload & Data prep
# -------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload KPI CSV", type=["csv"])
use_example = st.sidebar.checkbox("Use small example sample (if you don't have a file)", value=False)

if use_example and not uploaded:
    # create small synthetic example
    rng = pd.date_range("2022-01-01", periods=36, freq="MS")
    def mk(id_, name):
        vals = np.abs(np.round(np.random.normal(loc=1000, scale=200, size=len(rng)),2))
        return pd.DataFrame({
            "account_id":[id_]*len(rng),
            "english_name":[name]*len(rng),
            "dealer_code":["80475"]*len(rng),
            "year":rng.year,
            "month":rng.month,
            "monthly_value":vals,
            "date":rng
        })
    df_raw = pd.concat([mk("UNITS","Units Sold"), mk("REV","Revenue $"), mk("GP","Gross Profit")], ignore_index=True)
    st.info("Using synthetic example dataset (3 KPIs, 36 months).")
elif uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        # ensure required cols
        req = {"account_id","english_name","dealer_code","year","month","monthly_value"}
        if not req.issubset(set(df_raw.columns)):
            st.error(f"Uploaded CSV must contain columns: {sorted(req)}")
            st.stop()
        df_raw["date"] = pd.to_datetime(dict(year=df_raw["year"].astype(int), month=df_raw["month"].astype(int), day=1))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.info("Upload a KPI CSV on the left or check 'Use small example sample'.")
    st.stop()

# Ensure continuity and recompute YTD
with st.spinner("Cleaning and aligning monthly data..."):
    df_cont = ensure_monthly_continuity(df_raw)
    # ensure monthly_value exists and numeric
    if "monthly_value" in df_cont.columns:
        df_cont["monthly_value"] = pd.to_numeric(df_cont["monthly_value"], errors="coerce")
    else:
        st.error("monthly_value column missing.")
        st.stop()
    df_cont = recompute_ytd(df_cont)
    pivot = build_pivot(df_cont)

# -------------------------
# Sidebar: KPI selection + forecast
# -------------------------
st.sidebar.header("Forecast")
names_map = df_cont.set_index("account_id")["english_name"].to_dict()
kpi_choice_by_name = st.sidebar.selectbox("Choose KPI (by english_name)", sorted(df_cont["english_name"].unique()))
# map english_name -> account_id (pick first if duplicates)
kpi_choices = df_cont[df_cont["english_name"] == kpi_choice_by_name]["account_id"].unique()
kpi_account = st.sidebar.selectbox("Account ID", kpi_choices)

steps = st.sidebar.number_input("Forecast horizon (months)", min_value=1, max_value=12, value=3, step=1)

# -------------------------
# Forecast panel
# -------------------------
st.header("KPI Forecast")
col1, col2 = st.columns([2,1])

with col1:
    st.subheader(f"Historical & {steps}-month Forecast for: {kpi_choice_by_name} ({kpi_account})")
    series = df_cont[df_cont["account_id"] == kpi_account].set_index("date")["monthly_value"].dropna()
    if series.empty:
        st.warning("No historical monthly_value available for this KPI.")
    else:
        with st.spinner("Training SARIMAX (small grid search)..."):
            fc, ci, info = forecast_kpi(series, steps=steps)
        # plot history + forecast (plotly)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="History"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode="lines+markers", name="Forecast"))
        if ci is not None:
            fig.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:,1], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:,0], mode='lines', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,100,80,0.15)', showlegend=False))
        fig.update_layout(xaxis_title="Date", yaxis_title="Monthly Value")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Model info:", info)

        # table
        fc_df = pd.DataFrame({"date": fc.index, "forecast": fc.values})
        if ci is not None:
            fc_df["lo95"] = ci.iloc[:,0].values
            fc_df["hi95"] = ci.iloc[:,1].values
        st.dataframe(fc_df)

        # Export forecast
        excel_bytes = to_excel_bytes(fc_df)
        st.download_button("Download forecast (Excel)", data=excel_bytes, file_name=f"forecast_{kpi_account}.xlsx")

with col2:
    st.subheader("Quick stats")
    if not series.empty:
        st.metric("Last observed", f"{series.iloc[-1]:,.2f}")
        st.metric("Mean (history)", f"{series.mean():,.2f}")
        st.metric("Std (history)", f"{series.std():,.2f}")
    st.write("Data summary (per KPI counts):")
    counts = df_cont.groupby("account_id")["monthly_value"].count().sort_values(ascending=False)
    st.write(counts.head(10))

# -------------------------
# Correlation panel
# -------------------------
st.header("KPI Correlation Matrix")
with st.expander("Show / Recompute correlation matrix"):
    st.write("Correlation computed on monthly values (forward/backfill small gaps).")
    corr_fill = pivot.ffill().bfill()
    corr = corr_fill.corr()
    # heatmap
    fig_corr = px.imshow(corr, labels=dict(x="KPI (account_id)", y="KPI (account_id)", color="corr"),
                         x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                         title="KPI Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)
    # Top correlated for the selected kpi_account
    if kpi_account in corr.index:
        st.write(f"Top correlations for **{kpi_choice_by_name} ({kpi_account})**")
        s = corr[kpi_account].drop(labels=[kpi_account]).dropna()
        top_pos = s.sort_values(ascending=False).head(5)
        top_neg = s.sort_values(ascending=True).head(5)
        st.write("Top positive correlations:")
        st.write(top_pos.to_frame(name="corr"))
        st.write("Top negative correlations:")
        st.write(top_neg.to_frame(name="corr"))

# -------------------------
# What-If simulator
# -------------------------
st.header("What-If Simulator")
st.write("Pick a driver KPI, apply a % change, and see instant elastic impacts on top dependent KPIs (estimated by log-log OLS).")

colA, colB = st.columns(2)
with colA:
    driver_name = st.selectbox("Driver KPI (english_name)", sorted(df_cont["english_name"].unique()), key="driver_name")
    driver_accounts = df_cont[df_cont["english_name"] == driver_name]["account_id"].unique()
    driver_account = st.selectbox("Driver account_id", driver_accounts)
    pct_change = st.slider("Change (%) for driver KPI", min_value=-90, max_value=500, value=10, step=1)

with colB:
    n_dependents = st.number_input("Number of dependent KPIs to show", min_value=1, max_value=10, value=5)
    choose_base_month = st.date_input("Base month for instantaneous impact (choose last known month)", value=df_cont["date"].max())

# Build elasticities using top correlated KPIs as candidate drivers
with st.spinner("Estimating elasticities (log-log) ..."):
    # pick top correlated KPIs to be drivers/targets pool
    if driver_account not in pivot.columns:
        st.error("Driver KPI not present in pivot table (no data).")
    else:
        corr_series = corr[driver_account].drop(labels=[driver_account]).dropna()
        candidate_accounts = list(corr_series.sort_values(ascending=False).head(20).index)
        # We'll compute elasticity of each candidate target wrt the driver (single-driver log-log)
        elasticities = {}
        for tgt in candidate_accounts:
            el = compute_elasticities_loglog(pivot.fillna(method="ffill").fillna(method="bfill"), tgt, [driver_account])
            # el is {driver_account: coef} or {}
            if driver_account in el:
                elasticities[tgt] = el[driver_account]
        # pick top dependents by abs(elasticity)
        if not elasticities:
            st.warning("Not enough data to compute elasticities for dependents.")
            top_dependents = []
        else:
            top_dependents = sorted(elasticities.items(), key=lambda x: -abs(x[1]))[:n_dependents]

# Display base values and scenario
st.write("### Instant Impact (single-month algebraic estimate)")
base_month = pd.to_datetime(choose_base_month).replace(day=1)
base_row = pivot.reindex([base_month]).iloc[0] if base_month in pivot.index else pivot.ffill().iloc[-1]
# compute new values
pct = pct_change / 100.0
scenario_row = base_row.copy()
for tgt, el in top_dependents:
    # percent change in target ≈ elasticity * percent change in driver
    delta = el * pct
    scenario_row[tgt] = base_row[tgt] * (1 + delta)

# show table of changes
if top_dependents:
    out = []
    for tgt, el in top_dependents:
        before = base_row.get(tgt, np.nan)
        after = scenario_row.get(tgt, np.nan)
        out.append({
            "account_id": tgt,
            "english_name": df_cont[df_cont["account_id"]==tgt]["english_name"].iloc[0] if not df_cont[df_cont["account_id"]==tgt].empty else tgt,
            "elasticity": el,
            "before": before,
            "after": after,
            "abs_change": (after - before),
            "pct_change": (after - before) / (before if before else np.nan)
        })
    out_df = pd.DataFrame(out).round(4)
    st.dataframe(out_df)
    # bar chart of pct changes
    fig_bar = px.bar(out_df, x="english_name", y="pct_change", title=f"Percent change in dependents when {driver_name} changes by {pct_change}%")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No dependent KPIs to show — try a different driver or upload more data.")

st.markdown("---")
st.caption("Built with Streamlit • SARIMAX forecasts • Correlation & log-log elasticities for quick what-if analysis.")
st.caption("Thank You Scaletech.xyz")

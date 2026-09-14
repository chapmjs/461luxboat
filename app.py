"""
SCM 461 -- Throughput & Due-Date Quoting Calculator
Built on Iravani, "Operations Engineering and Management," Sec. 2.2
(the LuxBoat / Joseph worked example)

Run locally:            streamlit run app.py
Deploy with no install:  push this file + requirements.txt to a GitHub repo
                          (github.com web editor is enough -- no local git needed),
                          then deploy free at share.streamlit.io pointing at the repo.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Throughput & Due-Date Quoting", layout="wide")

st.title("Throughput & Due-Date Quoting Calculator")
st.caption("SCM 461 · built on Iravani Ch. 2.2 -- the LuxBoat / Joseph example")

# ---------- Sample data: LuxBoat, Table 2.1 ----------
LUXBOAT_SAMPLE = pd.DataFrame({
    "boat_no": [f"B{n:02d}" for n in range(1, 31)],
    "timestamp": pd.to_datetime([
        "2024-09-01 10:00", "2024-09-02 18:30", "2024-09-03 06:00", "2024-09-04 22:00",
        "2024-09-06 12:30", "2024-09-07 18:00", "2024-09-09 07:00", "2024-09-10 23:00",
        "2024-09-13 00:00", "2024-09-14 20:00", "2024-09-16 05:30", "2024-09-18 01:30",
        "2024-09-19 15:00", "2024-09-21 14:00", "2024-09-23 15:00", "2024-09-25 12:00",
        "2024-09-27 01:30", "2024-09-28 07:30", "2024-09-29 15:30", "2024-10-01 02:00",
        "2024-10-02 12:00", "2024-10-04 15:00", "2024-10-06 15:00", "2024-10-08 08:30",
        "2024-10-10 00:00", "2024-10-11 12:00", "2024-10-12 19:00", "2024-10-14 07:00",
        "2024-10-16 00:00", "2024-10-17 10:00",
    ]),
})

st.sidebar.header("1. Production completion data")
data_source = st.sidebar.radio(
    "Data source", ["LuxBoat sample data (Table 2.1)", "Upload your own CSV"]
)

if data_source == "Upload your own CSV":
    st.sidebar.caption("CSV needs one column of completion timestamps (any parseable date/time format).")
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        ts_col = st.sidebar.selectbox("Which column holds the timestamp?", df.columns)
        df["timestamp"] = pd.to_datetime(df[ts_col])
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        st.info("Upload a CSV in the sidebar, or switch to the LuxBoat sample data to see the tool run.")
        st.stop()
else:
    df = LUXBOAT_SAMPLE.copy()

st.sidebar.header("2. What are you quoting?")
b = st.sidebar.number_input("Units in the order (b)", min_value=1, value=25, step=1)
confidence = st.sidebar.slider("Desired confidence level (%)", 50, 99, 90, step=1)
hours_per_day = st.sidebar.number_input("Operating hours per day", min_value=1, max_value=24, value=24)

# ---------- Core calculations (Iravani Ch. 2.2) ----------
df = df.sort_values("timestamp").reset_index(drop=True)
n = len(df)
span_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
throughput_rate = (n - 1) / span_days if span_days > 0 else np.nan

inter_times_hours = df["timestamp"].diff().dt.total_seconds().dropna() / 3600
mean_x = inter_times_hours.mean()
var_x = inter_times_hours.var(ddof=1)
rho1 = inter_times_hours.autocorr(lag=1)

mu_b = b * mean_x
sigma2_b = ((1 + rho1) / (1 - rho1)) * b * var_x if rho1 not in (1, -1) else np.nan
sigma_b = np.sqrt(sigma2_b) if sigma2_b == sigma2_b else np.nan  # guard NaN

z_alpha = norm.ppf(confidence / 100)
t_due_hours = mu_b + z_alpha * sigma_b
t_due_days = t_due_hours / hours_per_day
naive_days = (b / throughput_rate) if throughput_rate else np.nan

# ---------- Display ----------
col1, col2, col3 = st.columns(3)
col1.metric("Throughput rate", f"{throughput_rate:.3f} units/day")
col2.metric("Mean inter-throughput time", f"{mean_x:.1f} hrs")
col3.metric("Lag-1 autocorrelation (ρ₁)", f"{rho1:.3f}")

st.subheader("Due-date quote")
c1, c2 = st.columns(2)
c1.metric("Naive estimate (b ÷ throughput rate)", f"{naive_days:.1f} days")
c2.metric(f"Quote at {confidence}% confidence (accounts for variability + dependence)",
          f"{t_due_days:.1f} days")

st.caption(
    f"Safety time built into this quote: {t_due_days - naive_days:.1f} days above the naive average -- "
    f"so the order ships on or before the quoted date about {confidence}% of the time, "
    "**if the assumptions below hold.**"
)

st.subheader("Inter-throughput times")
st.bar_chart(inter_times_hours.reset_index(drop=True))

st.subheader("Before you trust this number")
st.warning(
    """
This calculator can only compute the number -- it can't tell you whether the number is *right for this client*.
Before quoting this date, you still have to judge:

- **Is throughput actually stationary?** Look at the chart above -- is there a trend, or does it swing with the
  day of week or shift? A rising or falling trend breaks this whole calculation.
- **Is n large enough?** A handful of completions gives you a noisy mean, variance, and autocorrelation --
  small samples can make this quote look more confident than it should be.
- **Does this client's queue actually work like LuxBoat's?** First-come-first-served, no expediting, no rush
  orders cutting the line -- if that's not true here, the math still runs, but the answer is wrong.

That judgment is the part this app can't do for you.
"""
)

with st.expander("How this is calculated (Iravani Ch. 2.2)"):
    st.markdown(r"""
- **Throughput rate**: $TH = \dfrac{N_g}{t}$ -- good units produced per unit time.
- **Inter-throughput time**: the time between two consecutive completions, $X_i$.
- **Due date** for $b$ future units, assuming stationary throughput:

$$\mu_b = b\bar{X} \qquad \sigma_b^2 = \left(\frac{1+\rho_1}{1-\rho_1}\right) b S^2$$

$$T_{due} = \mu_b + z_\alpha \sigma_b$$

where $z_\alpha$ is the standard normal value for the desired confidence level.
""")

# %%

import os
import warnings
import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import adelie as ad
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path

from utils import range_regex

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# %%
# --- Page Config & Title ---
st.set_page_config(layout="wide", page_title="Surrogate Index")
st.title("High-Dimensional Linear Surrogate Indices: GAIN Experiment")

# %%
def filter_expr(prox_names, k):
    prox_expr = "|".join(["^" + x for x in prox_names])
    return f"({prox_expr}){range_regex(1, k)}$"

# @st.cache_data
def load_gain_data(yname, expr, data_path):
    """Load and process GAIN data based on user selections."""
    data = pd.read_csv(data_path)
    W = data['e']

    P = data.filter(regex=expr, axis=1)
    outcome_cols = [f'{yname}{i}' for i in range(33, 37)]
    Y = data[outcome_cols].mean(axis=1)
    return P, Y, W, data

# @st.cache_data
def calculate_tau_p(_W, _P):
    """Calculates the treatment effect on the proxies (Step 1)."""
    # Using np.c_ to create the design matrix [1, W]
    design_matrix = np.c_[np.ones_like(_W), _W.values]
    # np.linalg.lstsq returns a tuple, the first element is the solution
    res = np.linalg.lstsq(design_matrix, _P, rcond=None)[0]
    tau_p = res[1]  # The second row corresponds to the coefficient of W
    return tau_p

# %%
# --- Sidebar for User Controls ---
st.sidebar.header("Parameters")

# Parameter: Number of surrogate periods (k)
k = st.sidebar.slider(
    "1. Number of Surrogate Periods (k)",
    min_value=1, max_value=32, value=32,
    help="How many time periods (quarters) to include as Proxy outcomes."
)

# Parameter: Surrogate variables
all_proxies = ["aid", "tcedd", "ptcedd", "ymw"]
prox_names = st.sidebar.multiselect(
    "2. Surrogate Variables (Proxies)",
    options=all_proxies,
    default=["aid", "tcedd", "ymw"],
    help="Which short-term outcomes to use as surrogates."
)


outcome_choice = st.sidebar.radio(
    "3. Outcome Variable",
    options=["ymw", "tcedd", "aid"],
    index = 1,
    help="Which outcome to use as the long-term metric"
)

outcome_mapping = {
    "ymw": "Employed",
    "tcedd": "Earnings",
    "aid": "AFDC Indicator"
}

outcome_name = outcome_mapping[outcome_choice]

# Parameter: Regression technique
model_choice = st.sidebar.radio(
    "4. Surrogate Index Model",
    options=["OLS", "Ridge", "Lasso"],
    index=1, # Default to Ridge
    help="Which regression method to use to build the surrogate index (i.e., estimate alpha)."
)

covaradjust = st.sidebar.radio(
    "4. Covariate Adjustment",
    options=["Unadjusted", "Adjusted"],
    index=0,
    help="Covariate adjustment in long-term model"
)


######################################################################
######################################################################
######################################################################

# --- Main Application Logic ---
if not prox_names:
    st.warning("Please select at least one surrogate variable from the sidebar.")
    st.stop()


expr = filter_expr(prox_names, k)

# Load data based on sidebar selections
P, Y, W, data = load_gain_data(outcome_choice, expr, "raw/quarterly.csv")

covariate_list = [
    "grew1", "gepop1", "tcprn1", "tcprn2", "tcprn3", "tcprn4",
    "tcprn5", "tcprn6", "tcprn7", "tcprn8", "tcprn9", "tcprn10",
    "tcpp1", "tcpp2", "tcpp3", "tcpp4", "tcpp5", "tcpp6",
    "tcpp7", "tcpp8", "tcpp9", "tcpp10",
    "paid4", "paid3", "paid2", "paid1",
    "adcpc4", "adcpc3", "adcpc2", "adcpc1",
    "padcpc1", "padcpc2", "padcpc3", "padcpc4",
    "xsexf", "xhsdip", "x1chld", "single", "dumkids",
    "xchld05", "grd1720", "grade16", "grd1315", "grade12", "grde911",
    "white", "hisp", "black", "age", "agesq"
]


if covaradjust == "Adjusted":
    X = data[covariate_list]
    design_matrix_tau = np.c_[np.ones_like(W), W, X]
    ols_model = sm.OLS(Y, design_matrix_tau).fit(cov_type="HC1")
    tau_lt, se_lt = ols_model.params[1] , ols_model.bse[1]
else:
    design_matrix_tau = np.c_[np.ones_like(W), W]
    ols_model = sm.OLS(Y, design_matrix_tau).fit(cov_type="HC1")
    tau_lt, se_lt = ols_model.params[1] , ols_model.bse[1]

# Split data for ML models (not used by OLS on full data, but needed for Ridge/Lasso)
P_train, P_test, Y_train, Y_test = train_test_split(
    P, Y, test_size=0.1, random_state=42
)
# Convert to Fortran-contiguous array as recommended by adelie
P_train_f = np.asfortranarray(P_train.astype(float))
P_test_f = np.asfortranarray(P_test.astype(float))

# --- Display Results ---

# Column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Summary")
    st.metric("Total Observations", len(data))
    st.markdown(f"T: {W.sum()}, C: {(1-W).sum()}")
    st.metric("Number of Proxy Variables", P.shape[1])
    st.markdown(f"selection expr: `{expr}`")

with col2:
    st.metric("Primary Outcome", outcome_name)
    st.markdown(f"Average of `{outcome_choice}33-36`")
    st.subheader("Benchmark: Long-term Effect")
    st.metric(
        "Direct Average Treatment Effect (ATE)",
        f"{tau_lt:,.4f}",
        help="The simple difference in average long-term outcomes between the treated and control groups in the last 4 quarters"
    )


st.subheader(f" Surrogate Analysis using `{model_choice}`")
# --- Step 1: Estimate Tau_p (Effect of W on P) ---
# This step is the same for all models
tau_p = calculate_tau_p(W, P)

# --- Step 2: Estimate Alpha (Effect of P on Y) using chosen model ---
alpha_surr = None
model_info = {}

if model_choice == "OLS":
    # Fit OLS on the full dataset
    design_matrix_alpha = sm.add_constant(P)
    ols_model = sm.OLS(Y, design_matrix_alpha).fit()
    alpha_surr = ols_model.params[1:] # Exclude constant
    model_info['In-Sample R²'] = f"{ols_model.rsquared:.3f}"

elif model_choice in ["Ridge", "Lasso"]:
    model = ad.GroupElasticNet(solver = "cv_grpnet")
    alp = 0 if model_choice == "Ridge" else 1

    # Fit model on training data
    model.fit(P_train_f, Y_train, alpha=alp, min_ratio=1e-10, progress_bar=False)
    alpha_surr = model.state_.betas[-1].toarray().squeeze()

    # Evaluate on test data
    y_pred = model.predict(P_test_f)
    r2 = r2_score(Y_test, y_pred.squeeze())
    model_info['Out-of-Sample R²'] = f"{r2:.3f}"
    if model_choice == "Lasso":
        # Add sparsity info for Lasso
        selected_features = np.sum(alpha_surr != 0)
        model_info['Proxies Selected'] = f"{selected_features} / {P.shape[1]}"


# --- Final Result Calculation and Display ---
if alpha_surr is not None:
    surrogate_ate = np.dot(tau_p, alpha_surr)

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("📈 Model Performance")
        for key, val in model_info.items():
            st.metric(key, val)

    with res_col2:
        st.subheader("Final Surrogate ATE")
        st.metric(
            f"Surrogate ATE ({model_choice})",
            f"{surrogate_ate:,.5f}",
            delta=f"{surrogate_ate - tau_lt:,.4f} vs Benchmark",
            delta_color="normal",
            help="The estimated long-term effect using the surrogate index method. Delta shows the difference from the direct benchmark effect."
        )

    # Optional: Show the alpha coefficients in an expander
    with st.expander("🔍 View Surrogate Index Weights (alpha)"):
        st.write("These are the weights assigned to each proxy variable to form the surrogate index.")
        alpha_df = pd.DataFrame({'proxy': P.columns, 'coef': alpha_surr}).set_index("proxy")
        st.dataframe(
            alpha_df.sort_values(by=['coef'], key=abs, ascending=False).style.format({'coef': "{:.8f}"})
        )

st.subheader("Treatment Effects Dynamics")

fig, ax = plt.subplots(figsize=(8, 4))
aggs= data.groupby("e")[[f"{outcome_choice}{i}" for i in range(1, 37)]].mean()

treatment_means = aggs.loc[1].values
control_means = aggs.loc[0].values


# plt.style.use('dark_background')

fig, axes = plt.subplots(2, 1, figsize=(12,9))
ax = axes[0]
ax.plot(range(1, 37), treatment_means, 'b-', label='Treatment')
ax.plot(range(1, 37), control_means, 'r-', label='Control')
ax.grid(True, alpha=0.3)
ax.axvline(x=k, color='y', linestyle='--', label='Proxy/Outcome Split')

# Highlight the outcome region with a subtle background
ax.axvspan(33, 36, alpha=0.2, color='orange')

# Add text annotation
ax.text(34.5, max(treatment_means[-4:]) + 0.01, 'Outcome\nPeriod',
        ha='center', va='bottom', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='orange', alpha=0.9))

ax.set_xlabel("Quarter")
ax.legend()

ax = axes[1]
ax.plot(range(1, 37), treatment_means - control_means, 'g-', label='Treatment - Control')
ax.axhline(y=tau_lt, color='r', linestyle='--', label='LT Effect')
ax.axhline(y=tau_lt -  1.96 * se_lt, color='r', linestyle=':', alpha=0.3)
ax.axhline(y=tau_lt +  1.96 * se_lt, color='r', linestyle=':', alpha=0.3)

ax.axhline(y=0, color='k', linestyle='solid')
ax.axhline(y=surrogate_ate, color='magenta', linestyle='-.', label='Surrogate Estimate')
ax.axhline(y=np.mean(treatment_means[:k] - control_means[:k]),
        color='c', linestyle='--', label='Extrapolation')
ax.axvline(x=k, color='y', linestyle='-.')


ax.axvspan(33, 36, alpha=0.2, color='orange')
ax.legend()
fig.suptitle(f'{outcome_name}')

fig.tight_layout()
st.pyplot(fig)

st.divider()

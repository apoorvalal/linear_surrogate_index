# %%
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import adelie as ad
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from utils import range_regex


# %%
def filter_expr(prox_names, k):
    """Generate regex filter expression for proxy variables"""
    prox_expr = "|".join(["^" + x for x in prox_names])
    return f"({prox_expr}){range_regex(1, k)}$"


def load_gain_data(yname, expr, data_path):
    """Load and process GAIN data based on selections."""
    data = pd.read_csv(data_path)
    W = data["e"]
    P = data.filter(regex=expr, axis=1)
    outcome_cols = [f"{yname}{i}" for i in range(33, 37)]
    Y = data[outcome_cols].mean(axis=1)
    return P, Y, W, data


def calculate_tau_p(W, P):
    """Calculate treatment effect on proxies (Step 1)."""
    design_matrix = np.c_[np.ones_like(W), W.values]
    res = np.linalg.lstsq(design_matrix, P, rcond=None)[0]
    tau_p = res[1]  # Coefficient of W
    return tau_p


# %%

def fit_surrogate_model(P, Y, model_choice="Ridge", z=10):
    """Fit surrogate model and return alpha coefficients."""
    if model_choice == "OLS":
        design_matrix_alpha = sm.add_constant(P).values
        ols_model = sm.OLS(Y, design_matrix_alpha).fit()
        alpha_surr = ols_model.params[1:]
        abs_alpha = np.abs(alpha_surr.values)  # Convert pandas Series to numpy array
        top_z_indices = np.argsort(abs_alpha)[-z:]
        # Create reduced design matrix (intercept + selected variables)
        selected_cols = [0] + [i+1 for i in top_z_indices]
        design_matrix_alpha_reduced = design_matrix_alpha[:, selected_cols]
        # Refit on selected variables
        ols_model_reduced = sm.OLS(Y, design_matrix_alpha_reduced).fit()

        # Create result vector of original size, zero out non-selected
        alpha_result = np.zeros_like(alpha_surr)
        alpha_result[top_z_indices] = ols_model_reduced.params[
            1:
        ]  # Exclude intercept from reduced model

        return alpha_result

    elif model_choice in ["Ridge", "Lasso"]:
        # Split for cross-validation
        P_train, P_test, Y_train, Y_test = train_test_split(
            P, Y, test_size=0.1, random_state=42
        )
        P_train_f = np.asfortranarray(P_train.astype(float))

        model = ad.GroupElasticNet(solver="cv_grpnet")
        alp = 0 if model_choice == "Ridge" else 1
        model.fit(
            P_train_f,
            Y_train,
            alpha=alp,
            min_ratio=1e-10,
            progress_bar=False,
        )
        alpha_surr = model.state_.betas[-1].toarray().squeeze()
        return alpha_surr

    else:
        raise ValueError(f"Unknown model choice: {model_choice}")
# expr = filter_expr(["ymw"], 10)
# P, Y_config, W_config, _ = load_gain_data("ymw", expr, root / "gain_raw/quarterly.csv")
# P.shape
# d, ind = fit_surrogate_model(P, Y_config, "OLS", 3)



def calculate_surrogate_ate(W, P, Y, model_choice="Ridge", z=10):
    """Calculate surrogate ATE for given configuration."""
    tau_p = calculate_tau_p(W, P)
    alpha_surr = fit_surrogate_model(P, Y, model_choice, z)
    surrogate_ate = np.dot(tau_p, alpha_surr)
    return surrogate_ate

# %%
# %%


# %%
# Data path - adjust as needed
data_path = "../gain_raw/quarterly.csv"

# Load base data for outcome calculation
data = pd.read_csv(data_path)
W = data["e"]

# %%
def run_surrogate(
    configurations,
    outcome_choice="tcedd",
    model_choice="Ridge",
    z = 10,
    plot_raw = True,
):
    mapping = {
        "tcedd": "Earnings",
        "ymw": "Employment",
    }
    outcome_cols = [f"{outcome_choice}{i}" for i in range(33, 37)]
    # long term outcome
    Y = data[outcome_cols].mean(axis=1)
    # Calculate benchmark long-term effect
    design_matrix_tau = np.c_[np.ones_like(W), W]
    ols_model = sm.OLS(Y, design_matrix_tau).fit(cov_type="HC1")
    tau_lt, se_lt = ols_model.params[1], ols_model.bse[1]
    # Define surrogate configurations
    # Calculate surrogate estimates for each configuration
    surrogate_results = []

    for config in configurations:
        expr = filter_expr(config["prox_names"], config["k"])
        P, Y_config, W_config, _ = load_gain_data(outcome_choice, expr, data_path)

        if P.shape[1] > 0:  # Ensure we have proxy variables
            surrogate_ate = calculate_surrogate_ate(W_config, P, Y_config, model_choice, z = z)
            surrogate_results.append(
                {
                    "name": config["name"],
                    "ate": surrogate_ate,
                    "color": config["color"],
                    "k": config["k"],
                }
            )
            print(f"{config['name']}: {surrogate_ate:.5f}")
        else:
            print(f"No proxy variables found for {config['name']}")

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Calculate treatment dynamics for plotting
    aggs = data.groupby("e")[[f"{outcome_choice}{i}" for i in range(1, 37)]].mean()
    treatment_means = aggs.loc[1].values
    control_means = aggs.loc[0].values
    treatment_effects = treatment_means - control_means

    if plot_raw:
      # Top panel: Treatment and control means
      ax = axes[0]
      ax.plot(range(1, 37), treatment_means, "b-", label="Treatment", linewidth=2)
      ax.plot(range(1, 37), control_means, "r-", label="Control", linewidth=2)
      ax.grid(True, alpha=0.3)
      ax.axvspan(33, 36, alpha=0.2, color="orange")
      ax.text(
          34.5,
          max(treatment_means[-4:]) + 0.01,
          "Outcome\nPeriod",
          ha="center",
          va="bottom",
          fontsize=10,
          bbox=dict(
              boxstyle="round,pad=0.2", facecolor="white", edgecolor="orange", alpha=0.9
          ),
      )
      ax.set_ylabel("Outcome Level")
      ax.legend()
      ax.set_title(f"Raw outcomes by group - {mapping[outcome_choice].upper()}")

    # Bottom panel: Treatment effects with surrogate estimates
    ax = axes[1]
    ax.plot(
        range(1, 37), treatment_effects, "g-", label="Treatment - Control", linewidth=2
    )

    ax.axhline(
        y=np.mean(treatment_effects[:12]),
        color="c",
        linestyle="--",
        label="Extrapolation",
    )

    # Add benchmark long-term effect
    ax.axhline(
        y=tau_lt, color="red", linestyle="--", linewidth=2, label="Benchmark LT Effect"
    )
    ax.axhline(y=tau_lt - 1.96 * se_lt, color="red", linestyle=":", alpha=0.5)
    ax.axhline(y=tau_lt + 1.96 * se_lt, color="red", linestyle=":", alpha=0.5)

    # Add surrogate estimates
    for result in surrogate_results:
        ax.axhline(
            y=result["ate"],
            color=result["color"],
            linestyle="-.",
            linewidth=2,
            label=f"Surrogate: {result['name']}",
        )

    # Add reference lines and shading
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.axvspan(33, 36, alpha=0.2, color="orange")

    # Add vertical lines for different k values used
    for result in surrogate_results:
        if result["k"] < 32:
            ax.axvline(x=result["k"], color=result["color"], linestyle=":", alpha=0.7)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Treatment Effect")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title(f"Effect estimates - {model_choice}")

    if not plot_raw:
      fig.delaxes(axes[0])
    fig.tight_layout()
    return fig, axes

# %%
configurations = [
    {
        "name": "16 qtrs lagged y only",
        "prox_names": ["ymw"],
        "k": 16,
        "color": "purple",
    },
    {
        "name": "32 qtrs lagged y only",
        "prox_names": ["ymw"],
        "k": 32,
        "color": "blue",
    },
    {
        "name": "32 qtrs all surrogates",
        "prox_names": ["aid", "tcedd", "ymw"],
        "k": 32,
        "color": "green",
    },
]

# %%
f, ax = run_surrogate(configurations, "ymw", "Ridge")
f.savefig("../results/ridge_gain.png", dpi=200)
# %%
f, ax = run_surrogate(configurations, "ymw", "OLS", z=4, plot_raw = False)
f.savefig("../results/ols_gain.png", dpi=200)
# %%
f, ax = run_surrogate(configurations, "ymw", "Lasso", plot_raw = False)
f.savefig("../results/lasso_gain.png", dpi=200)
# %%


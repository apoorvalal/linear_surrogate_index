# %%
import os
import warnings
from utils import range_regex

import numpy as np
import pandas as pd

import adelie as ad

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import statsmodels.api as sm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Make results directory
os.makedirs("results", exist_ok=True)



# %%
def load_gain_data(yname="ymw", prox_names= ["aid", "tcedd", "ptcedd", "ymw"], k = 32,
        data_path="../gain_raw/quarterly.csv"):
    """
    Load GAIN data and prepare for analysis

    Returns:
    --------
    P : ndarray
        Proxy matrix (quarters 1-32)
    Y : ndarray
        Long-term outcome (average of quarters 33-36)
    W : ndarray
        Treatment indicator
    data : DataFrame
        Full dataset for additional analyses
    """
    # Load data
    data = pd.read_csv(data_path)
    # Extract treatment indicator
    W = data['e']
    # extract proxies based on prox filters and periods
    prox_expr = "|".join(["^"+x for x in prox_names])
    P = data.filter(regex = f"({prox_expr}){range_regex(1, k)}$", axis = 1)
    # Create long-term outcome (average of quarters 33-36)
    outcome_cols = [f'{yname}{i}' for i in range(k+1, 37)]
    Y = data[outcome_cols].mean(axis=1)
    print(f"Data loaded: {len(data)} observations")
    print(f"Treatment group: {W.sum()} observations")
    print(f"Control group: {(1-W).sum()} observations")
    print(f"Number of proxy variables: {P.shape[1]}")
    return P, Y, W, data

# %%
P, Y, W, data = load_gain_data(prox_names = ["aid", "tcedd", "ptcedd", "ymw"])

m = sm.OLS(Y, sm.add_constant(W))
print(m.fit(cov_type="HC1").summary().tables[1])
print(true_effect := Y[W==1].mean() - Y[W==0].mean())
# %%
print(P.shape, Y.shape, W.shape)
# sample split for ridge
P_train, P_test, Y_train, Y_test, W_train, W_test = train_test_split(P, Y, W, test_size=0.2)
P_train, P_test = np.asfortranarray(P_train.astype(float)), np.asfortranarray(P_test.astype(float))
P_train.shape, P_test.shape, W_train.shape, W_test.shape, Y_train.shape, Y_test.shape
# %% estimate treatment effects on short-run outcomes
res = np.linalg.lstsq(np.c_[np.ones_like(W), W.values], P, rcond=None)[0]
tau_p = res[1] # first column is interecept, second column is treatment effect
# full OLS surrogate index
alpha = np.linalg.lstsq(np.c_[np.ones(P.shape[0]), P], Y)[0][1:]
np.dot(tau_p, alpha)
# %%
sorted_indices = np.argsort(np.abs(alpha))
print(selected_surrogates := sorted_indices[-1:])
np.dot(tau_p[selected_surrogates], alpha[selected_surrogates])
# %%
surr_mod = ad.GroupElasticNet()
# alpha = 0 : ridge
# alpha = 1 : lasso
alp = 1
mod_class = 'ridge' if alp == 0 else 'lasso'
surr_mod.fit(P_train, Y_train, lmda_path = np.logspace(-4, 3, 100), alpha = alp)
yhatmat = surr_mod.predict(P_test)
r2vec = np.apply_along_axis(lambda yhat: r2_score(Y_test, yhat), axis=1, arr=yhatmat)
lam_path = surr_mod.lambda_
plt.plot(-np.log(lam_path), r2vec, linestyle="None", marker=".")
plt.axhline(r2vec.max(), color="red", linestyle="--", label=f"max $R^2$ = {r2vec.max():.2f}")
plt.title(f"Surrogate Index Predictive Fit \nOut-of-sample $R^2$ over $-\\log(\\lambda)$\n {mod_class}")
plt.xlabel(r"$-\log(\lambda$)")
plt.ylabel(r"$R^2$")
plt.legend()
plt.savefig(f"surrind_r2_{mod_class}.png", dpi = 300)
plt.show()

# %%
alp = 0
surr_mod = ad.GroupElasticNet(solver = "cv_grpnet")
surr_mod.fit(P_train, Y_train, alpha = alp, min_ratio=1e-10)
surr_mod.score(P_test, Y_test)
surr_coefs = surr_mod.state_.betas[-1]
# %%
alpha_reg = surr_coefs.toarray().squeeze()
np.dot(tau_p, alpha_reg)
# %%

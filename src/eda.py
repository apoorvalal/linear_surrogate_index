# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('../gain_raw/quarterly.csv')

# # %% four counties - MEEX
# counties = ['alameda', 'river', 'sandiego', 'la']
# data[counties].sum().sum(), data.shape[0]
# Extract treatment indicator
W = data['e']
yname = "tcedd" # employment
# yname = "ymw" # employment
Y = data.filter(regex=f"(^{yname})(3[3-6])$",axis=1).mean(axis=1)
# Y = data[yname].values

P = data.filter(regex=f"(^{yname})([1-9]|[2-2][0-9]|1[0-9]|3[0-2])$",axis=1)
# %%
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

X = data[covariate_list].assign(
    intercept=1.0,)

# %%
def residualize(X, Y):
    m = np.linalg.lstsq(X, Y, rcond=None)[0]
    return Y - X @ m

ytilde = residualize(X=X, Y=Y)
Ptilde = residualize(X=X, Y=P)
ytilde[W == 1].mean() - ytilde[W == 0].mean()

# %%

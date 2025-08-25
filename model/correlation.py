# correlation.py
import pandas as pd
import numpy as np
import statsmodels.api as sm

def compute_elasticities_loglog(wide_df, target_id, driver_ids):
    eps = 1e-6
    sub = wide_df[[target_id] + driver_ids].dropna()
    if sub.shape[0] < max(12, len(driver_ids)*3):
        return {}
    X = np.log(sub[driver_ids].replace(0, eps))
    y = np.log(sub[target_id].replace(0, eps))
    X_const = sm.add_constant(X)
    try:
        model = sm.OLS(y, X_const).fit()
        return model.params.drop("const", errors="ignore").to_dict()
    except:
        return {}

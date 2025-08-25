# forecast.py
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def small_sarimax_grid_search(y, seasonal_periods=12, max_p=2, max_q=2, max_P=1, max_Q=1):
    y = y.dropna()
    if len(y) < 18:
        return None, None
    best_aic = np.inf
    best_order, best_seas = None, None
    pdq = [(p,d,q) for p in range(max_p+1) for d in (0,1) for q in range(max_q+1)]
    PDQ = [(P,D,Q,seasonal_periods) for P in range(max_P+1) for D in (0,1) for Q in range(max_Q+1)]
    for order in pdq:
        for seas in PDQ:
            try:
                mod = SARIMAX(y, order=order, seasonal_order=seas,
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                if res.aic < best_aic:
                    best_aic, best_order, best_seas = res.aic, order, seas
            except:
                continue
    return best_order, best_seas

def seasonal_naive_forecast(y, steps=3, m=12):
    y = y.dropna()
    if len(y) < m:
        return pd.Series([y.iloc[-1]]*steps, index=pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS"))
    last_season = y.iloc[-m:]
    vals = np.tile(last_season.values, int(np.ceil(steps/m)))[:steps]
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    return pd.Series(vals, index=idx)

def forecast_kpi(y, steps=3):
    y = y.dropna()
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    order, seas = small_sarimax_grid_search(y)
    if order is None:
        fc = seasonal_naive_forecast(y, steps)
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
    except:
        fc = seasonal_naive_forecast(y, steps)
        return fc, None, ("seasonal_naive", None, None)

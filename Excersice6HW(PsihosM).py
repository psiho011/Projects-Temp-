# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:12:12 2025

@author: mpsih
"""

#%%
#Problem 1
print("The data is representing the PE ratios at certain points in time between 1861 and 2019, It also refers to the S&P500 or the market as what the PE ratio is for")
print("\nIt is a non stationary model.")
print("\nIt is not able to use a ARMA(p,q) model")
#%%
#Problem 2
# Plot ACF & PACF from cape_data.csv

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#1) Load & prep
path = "cape_data.csv"  # update to whatever the file is
df = pd.read_csv(path)

#normalize headers on spaces and underscores
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

#build monthly datetime index from the CSV File
if {"year", "month"}.issubset(df.columns):
    df["day"] = 1
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
else:
    raise ValueError("Need either (year, month) columns or a 'date' column.")

#choose the TR-CAPE column using for loops and if statements
val_col = next((c for c in df.columns if "cape" in c), None)
if val_col is None:
    raise ValueError("Couldn't find a TR-CAPE column (one containing 'cape').")

y = (
    df.sort_values("date")
      .set_index("date")[val_col]
      .astype(float)
      .dropna()
)

#2) Choose which series to plot 
series = y


#3) ACF & PACF plots 
plt.figure(figsize=(8,5))
plot_acf(series, lags=60)
plt.title("ACF")
plt.show()

plt.figure(figsize=(8,5))
plot_pacf(series, lags=60, method="ywm")
plt.title("PACF")
plt.show()

#%%
#Problem 3

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# 1) Subset to pre-2010
y_pre2010 = y.loc[:'2009-12-31']

# 2) Fit a parsimonious model on levels with one difference
model = ARIMA(y_pre2010, order=(0,1,1)).fit()

# 3) Residuals
resid = model.resid

# 4) Plots: residuals + ACF of residuals
plt.figure(figsize=(9,4))
plt.plot(resid)
plt.title("Residuals (pre-2010 fit)")
plt.xlabel("Date"); plt.ylabel("Residual")
plt.show()

plt.figure(figsize=(8,5))
plot_acf(resid.dropna(), lags=60)
plt.title("ACF of Residuals (pre-2010 fit)")
plt.show()
#%%
#Problem 4
# 1) In-sample fitted values (pre-2010)
#Alligned to the index
fitted_pre = model.fittedvalues  

# 2) Out-of-sample forecast from 2010-01 through end of sample
y_out = y.loc['2010-01-01':]
steps = len(y_out)
fc = model.get_forecast(steps=steps)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.05)  # 95% CI
fc_ci_lower = fc_ci.iloc[:, 0]
fc_ci_upper = fc_ci.iloc[:, 1]

# Ensure forecast index aligns to actual out-of-sample dates
fc_mean.index = y_out.index
fc_ci_lower.index = y_out.index
fc_ci_upper.index = y_out.index

# 3) Plot observed, fitted (pre-2010), and forecast + 95% CI
plt.figure(figsize=(11,6))
plt.plot(y, label='Observed', linewidth=1.2)
plt.plot(fitted_pre, label='Fitted (pre-2010)', linewidth=1.2)
plt.plot(fc_mean, label='Forecast (2010→end)', linewidth=1.4)

plt.fill_between(y_out.index, fc_ci_lower, fc_ci_upper, alpha=0.2, label='95% CI')

plt.axvline(pd.Timestamp('2010-01-01'), color='k', linestyle='--', linewidth=1, alpha=0.6)
plt.title('TR-CAPE: Observed, Pre-2010 Fit, and Post-2010 Forecast (with 95% CI)')
plt.xlabel('Date'); plt.ylabel('TR-CAPE')
plt.legend()
plt.tight_layout()
plt.show()
print("The forecast tracks the broad post-2010 trend but misses turning points, with uncertainty bands widening quickly. It’s reasonable to trust directionality only for ~6–12 months; beyond that, error bands and potential regime shifts make it unreliable.")
#%%
#Problem 5
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

#Put the AR(1) on the ARMIA timer series
ar1 = ARIMA(y_pre2010, order=(1,1,0)).fit()

# 2) Residual diagnostics: Plot all of the residuals
resid_ar1 = ar1.resid

plt.figure(figsize=(9,4))
plt.plot(resid_ar1)
plt.title("Residuals — AR(1) fit on pre-2010 (ARIMA(1,1,0))")
plt.xlabel("Date"); plt.ylabel("Residual")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
plot_acf(resid_ar1.dropna(), lags=60)
plt.title("ACF of Residuals — AR(1) fit (pre-2010)")
plt.tight_layout(); plt.show()

#Forecast and set up 
y_out = y.loc['2010-01-01':]
steps = len(y_out)

fc_ar1 = ar1.get_forecast(steps=steps)
fc_ar1_mean = fc_ar1.predicted_mean
fc_ar1_ci = fc_ar1.conf_int(alpha=0.05)
fc_ar1_mean.index = y_out.index
fc_ar1_ci.index = y_out.index

plt.figure(figsize=(11,6))
plt.plot(y, label='Observed', linewidth=1.0)
plt.plot(ar1.fittedvalues, label='Fitted (pre-2010, AR(1))', linewidth=1.2)
plt.plot(fc_ar1_mean, label='Forecast (2010→end, AR(1))', linewidth=1.4)
plt.fill_between(y_out.index, fc_ar1_ci.iloc[:,0], fc_ar1_ci.iloc[:,1], alpha=0.2, label='95% CI (AR(1))')
plt.axvline(pd.Timestamp('2010-01-01'), color='k', linestyle='--', linewidth=1, alpha=0.6)
plt.title('TR-CAPE: AR(1) Pre-2010 Fit & Post-2010 Forecast')
plt.xlabel('Date'); plt.ylabel('TR-CAPE'); plt.legend(); plt.tight_layout(); plt.show()

# AR vs ARMIA(1) comparison
# Recompute the 0,1,1 forecasts here to ensure both still exist
fc_011 = model.get_forecast(steps=steps)
fc_011_mean = fc_011.predicted_mean
fc_011_mean.index = y_out.index

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

rmse_011 = rmse(y_out, fc_011_mean)
rmse_ar1 = rmse(y_out, fc_ar1_mean)

print(f"Out-of-sample RMSE — ARIMA(0,1,1): {rmse_011:.4f}")
print(f"\nOut-of-sample RMSE — AR(1) [ARIMA(1,1,0)]: {rmse_ar1:.4f}")
print("\n\nthe RMSE for your estimated model from 3 and for the AAAA(11) model here. ")
print("\nWhich performs better? Is this a surprising result? Why?")





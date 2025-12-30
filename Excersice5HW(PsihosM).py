# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:35:07 2025

@author: maxpsihos
"""
#%%
import pandas as pd
#Problems 1

# File format: [YYYYMMDD] [excess_return_decimal]
bh = pd.read_csv(
    'berkshire.txt',
    sep='\t',
    header=None,
    names=['date_raw', 'bh_excess'],
    dtype={'date_raw': str}
)

# Split YYYYMMDD into Year, Month, Day format
bh['year']  = bh['date_raw'].str.slice(0, 4).astype(int)   # YYYY
bh['month'] = bh['date_raw'].str.slice(4, 6).astype(int)   # MM
bh['day']   = bh['date_raw'].str.slice(6, 8).astype(int)   # DD

# Monthly key
bh['ym'] = pd.PeriodIndex(year=bh['year'], month=bh['month'], freq='M')

# Average daily bh_excess within each month
bh = (bh.groupby('ym', as_index=False)['bh_excess']
        .mean()
        .sort_values('ym'))



#FF3 Market Excess Return 
# File format: [Year] [Month] [Mkt-RF] [SMB] [HML] [RF]  (percent units)
ff3 = pd.read_csv(
    'FF3factors.txt',
    sep='\t',
    header=None,
    names=['year', 'month', 'mkt_excess_pct', 'SMB', 'HML', 'RF']
)


# Make a monthly key
ff3['ym'] = pd.PeriodIndex(year=ff3['year'].astype(int),
                           month=ff3['month'].astype(int),
                           freq='M')
#Drop 0s
bh = bh[bh['bh_excess'] != 0]
# Convert percent to decimal and keep only the market excess
ff3_mkt = ff3[['ym', 'mkt_excess_pct']].copy()
ff3_mkt['mkt_excess'] = ff3_mkt['mkt_excess_pct'] / 100.0
ff3_mkt = ff3_mkt.drop(columns='mkt_excess_pct').sort_values('ym')


# merge on the month
merged = pd.merge(bh, ff3_mkt, on='ym', how='inner').sort_values('ym')


#   bh:       monthly Berkshire excess returns (decimal)
#   ff3_mkt:  monthly market excess return from FF3 (decimal)
#   merged:   both, aligned by month
print(bh.head())
print(ff3_mkt.head())
print(merged.head())
#%%
#Problems 2
import scipy.stats as stats
import numpy as np
#Average arithmetic monthly Berkshire excess return
bh_avg_monthly = bh['bh_excess'].mean()
print(f"Average monthly Berkshire excess return: {bh_avg_monthly:.4%}")

#geometric average monthly (compounded) ---
bh_geo_monthly = (1.0 + bh['bh_excess']).prod()**(1.0/len(bh)) - 1.0
print(f"Geometric avg monthly Berkshire excess return: {bh_geo_monthly:.4%}")

# --- Optional: annualized versions ---
# Arithmetic annualization (rule of thumb)
#bh_avg_annual_arith = bh_avg_monthly * 12
#print(f"Arithmetic annualized (≈): {bh_avg_annual_arith:.4%}")

# Geometric annualization (compounded)
#bh_geo_annual = (1.0 + bh_geo_monthly)**12 - 1.0
#print(f"Geometric annualized: {bh_geo_annual:.4%}")

#Sample stats
n = len(bh)
mean_excess = bh['bh_excess'].mean()
std_excess = bh['bh_excess'].std(ddof=1)
se = std_excess / np.sqrt(n)

print(f"Sample size: {n}")
print(f"Mean excess return: {mean_excess:.4%}")
print(f"Std. dev: {std_excess:.4%}")
print(f"Std. error: {se:.4%}")

#Hypothesis test 1: H0 mean = 0 
null_val1 = 0.0
t_stat1 = (mean_excess - null_val1) / se
df = n - 1
p_val1 = 2 * (1 - stats.t.cdf(abs(t_stat1), df))
ci1 = stats.t.interval(0.95, df, loc=mean_excess, scale=se)

print("\nTest 1: H0: mean = 0")
print(f"t-statistic: {t_stat1:.3f}, p-value: {p_val1:.4f}")
print(f"95% CI: [{ci1[0]:.4%}, {ci1[1]:.4%}]")

#Hypothesis test 2: H0 mean = 1% (0.01)
null_val2 = 0.01
t_stat2 = (mean_excess - null_val2) / se
p_val2 = 2 * (1 - stats.t.cdf(abs(t_stat2), df))
ci2 = stats.t.interval(0.95, df, loc=mean_excess, scale=se)
#Formatting our output to answer the questions provided in HW
print("\nTest 2: H0: mean = 1%")
print(f"t-statistic: {t_stat2:.3f}, p-value: {p_val2:.4f}")
print(f"95% CI: [{ci2[0]:.4%}, {ci2[1]:.4%}]")

#%%
#Problem 2b
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
#Set the degrees of freedom for problem
df = n - 1  
#create a function to find t stats of our two dataframes 
def plot_t_test(null_val, label):
    t_stat = (mean_excess - null_val) / se
    x = np.linspace(-4, 4, 500)
    y = stats.t.pdf(x, df)

    plt.figure(figsize=(10,6))
    plt.plot(x, y, label=f't-dist (df={df})')

    # Set up the critical values for the 95% CI
    crit = stats.t.ppf(0.975, df)
    #Plot both ways so we can see the entire trend of the graph here
    plt.axvline(crit, color='red', linestyle='--', label=f'Critical ±{crit:.2f}')
    plt.axvline(-crit, color='red', linestyle='--')

    # Put a blue line at the marked t statistic 
    plt.axvline(t_stat, color='blue', linewidth=2, label=f'Observed t = {t_stat:.2f}')

    # rejection regions shaded
    plt.fill_between(x, y, 0, where=(x > crit) | (x < -crit), color='red', alpha=0.3)

    plt.title(f"T-test for H0: mean = {null_val:.2%} ({label})")
    plt.legend()
    plt.show()

# Plot for both tests
plot_t_test(0.0, "Test 1 (H0 = 0)")
plot_t_test(0.01, "Test 2 (H0 = 1%)")
#%%
#Problem 3 
#Factor regressions

import numpy as np
from scipy.stats import t as tdist

# 1) Build factors in DECIMALS and merge with Berkshire monthly
fac = ff3[['ym','mkt_excess_pct','SMB','HML']].copy()
fac[['Mkt','SMB','HML']] = fac[['mkt_excess_pct','SMB','HML']] / 100.0
fac = fac.drop(columns='mkt_excess_pct')
#merge on bhk
df_reg = bh.merge(fac, on='ym', how='inner').dropna()
y = df_reg['bh_excess'].to_numpy().reshape(-1,1)

def ols_robust(y, X):
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    # (X'X)^(-1) X'y
    beta = XtX_inv @ (X.T @ y)   
    # residuals (n x 1)              
    u = y - X @ beta                         

    # s^2 (X'X)^(-1)
    s2      = float((u.T @ u) / (n - k))
    var_h   = s2 * XtX_inv
    se_h    = np.sqrt(np.diag(var_h))

    #(X'X)^(-1) (X' diag(u^2) X) (X'X)^(-1)
    # row-scale X by u_i, so Z'Z = X'diag(u^2)X
    Z       = X * u                                
    var_r   = XtX_inv @ (Z.T @ Z) @ XtX_inv
    se_r    = np.sqrt(np.diag(var_r))

    return beta.ravel(), se_h, se_r, n, k

def stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

def run(name, cols):
    X = np.column_stack([np.ones(len(df_reg)), df_reg[cols].to_numpy()])
    beta, se_h, se_r, n, k = ols_robust(y, X)
    dfree  = n - k
    t_h    = beta / se_h
    p_h    = 2 * (1 - tdist.cdf(np.abs(t_h), dfree))
    t_r    = beta / se_r
    p_r    = 2 * (1 - tdist.cdf(np.abs(t_r), dfree))

    print(f"\n{name} (n={n}, k={k})")
    print(f"{'var':<6} {'coef':>10} {'se':>10} {'t':>8} {'p':>8}   {'rob_se':>10} {'rob_t':>8} {'rob_p':>8}")
    labs = ['alpha'] + cols
    for i, lab in enumerate(labs):
        print(f"{lab:<6} {beta[i]:>10.6f} {se_h[i]:>10.6f} {t_h[i]:>8.2f} {p_h[i]:>8.4f}   ||   "
              f"{se_r[i]:>10.6f} {t_r[i]:>8.2f} {p_r[i]:>8.4f} {stars(p_r[i])}")
    return beta[0], p_r[0]  # return alpha and its robust p

# Run the 3 models
a_capm, p_capm = run("CAPM",['Mkt'])
a_2f,   p_2f   = run("2-factor",[ 'Mkt','SMB'])
a_ff3,  p_ff3  = run("FF3 (3-factor)", ['Mkt','SMB','HML'])

print(f"\nRobust alphas: CAPM={a_capm:.6f} (p={p_capm:.4f}), "
      f"2-factor={a_2f:.6f} (p={p_2f:.4f}), FF3={a_ff3:.6f} (p={p_ff3:.4f})")

print("What we notice about a is that as alpha shrinks or losses signifigance you go from CAPM to 2factor to FF3. It shows us that the strange return can be accounted for by its size and value rather then stock pricing")
print("If alpha stays large that shows persistant outpurformance of the ff3 structure")
#%%
#Problem 4
#R^2 and Adjusted R^2 for 2-factor vs FF3

import numpy as np

# Build factors in DECIMALS and merge with Berkshire monthly 
fac = ff3[['ym','mkt_excess_pct','SMB','HML']].copy()
fac[['Mkt','SMB','HML']] = fac[['mkt_excess_pct','SMB','HML']] / 100.0
fac = fac.drop(columns='mkt_excess_pct')

df_reg = bh.merge(fac, on='ym', how='inner').dropna()
y = df_reg['bh_excess'].to_numpy().reshape(-1,1)
n = len(df_reg)

def fit_r2(y, X):
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    SSE = float(u.T @ u)
    SST = float(((y - y.mean())**2).sum())
    R2 = 1 - SSE/SST
    adjR2 = 1 - (SSE/(n - k)) / (SST/(n - 1))

    # homoskedastic SEs & t for quick check 
    s2 = SSE/(n - k)
    SE = np.sqrt(np.diag(s2 * XtX_inv))
    tvals = (beta.ravel() / SE)
    return beta.ravel(), R2, adjR2, tvals

# 2-factor (Mkt, SMB)
X2  = np.column_stack([np.ones(n), df_reg[['Mkt','SMB']].to_numpy()])
b2, R2_2f, adj2_2f, t2 = fit_r2(y, X2)

# FF3 (Mkt, SMB, HML)
X3  = np.column_stack([np.ones(n), df_reg[['Mkt','SMB','HML']].to_numpy()])
b3, R2_ff3, adj2_ff3, t3 = fit_r2(y, X3)

print(f"FF3 R^2 = {R2_ff3:.4f}  -> {R2_ff3*100:.1f}% of variance explained")
print(f"Unexplained (idiosyncratic + alpha/noise) = {(1 - R2_ff3)*100:.1f}%")

print(f"\n2-factor:     R^2 = {R2_2f:.4f},  adj R^2 = {adj2_2f:.4f}")
print(f"FF3 (3-fact): R^2 = {R2_ff3:.4f}, adj R^2 = {adj2_ff3:.4f}")

# Did adding HML help? 
# last coefficient corresponds to HML in FF3
t_HML = t3[-1]  
improved = (adj2_ff3 > adj2_2f) and (t_HML > 1)
print(f"\nHML t-stat in FF3: {t_HML:.2f}")
print(f"Adjusted R^2 improved after adding HML? {'YES' if improved else 'NO'}")

#%%
#Problem 5

import numpy as np

# 1) Factors in DECIMALS + merge with Berkshire monthly
fac = ff3[['ym','mkt_excess_pct','SMB','HML']].copy()
fac[['Mkt','SMB','HML']] = fac[['mkt_excess_pct','SMB','HML']] / 100.0
fac = fac.drop(columns='mkt_excess_pct')

df_vd = bh.merge(fac, on='ym', how='inner').dropna()
# (n,)
y   = df_vd['bh_excess'].to_numpy()  
# (n,3)                        
Xf  = df_vd[['Mkt','SMB','HML']].to_numpy()    
# add intercept            
X = np.column_stack([np.ones(len(y)), Xf])                  

# 2) OLS for FF3 (alpha, betas)
beta = np.linalg.inv(X.T @ X) @ (X.T @ y)    
# factor loadings only                 
b = beta[1:]                                               

# 3) R^2 (how much variance the model explains aka its relationship to the model)
#Find R^2 using SSE and SST
yhat = X @ beta
SSE= np.sum((y - yhat)**2)
SST = np.sum((y - y.mean())**2)
R2   = 1 - SSE/SST
print(f"FF3 R^2: {R2:.4f}  -> {R2*100:.1f}% of Berkshire variance explained")
print(f"Unexplained (idiosyncratic): {(1-R2)*100:.1f}%\n")

# 4) Variance decomposition:
# Build  = Cov(factors), then contribution matrix C = (b b' @ Σ_X) / Var(y)
# 3x3 covariance of [Mkt, SMB, HML]
SigmaX = np.cov(Xf, rowvar=False, ddof=1)                   
var_y = np.var(y, ddof=1)
# elementwise product
C = (np.outer(b, b) * SigmaX) / var_y                  
# factor-by-factor
diag_shr = np.diag(C)
# should ≈ R^2                                         
total_shr = C.sum()                                           

names = ['Mkt','SMB','HML']
for nm, s in zip(names, diag_shr):
    print(f"{nm} variance contribution (diag): {s:.4f}  ({s*100:.1f}%)")
print(f"\nSum of ALL entries of C (≈ R^2): {total_shr:.4f}")
print(f"R^2 check: {R2:.4f}  |  Unexplained: {1-R2:.4f}")

#%%
#Problem 6
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
#merge regression together
reg = bh.merge(fac, on="ym", how="inner").dropna()
#use statsmodel to estimate
res_capm = smf.ols("bh_excess ~ Mkt", data=reg).fit()
res_2f   = smf.ols("bh_excess ~ Mkt + SMB", data=reg).fit()
res_ff3  = smf.ols("bh_excess ~ Mkt + SMB + HML", data=reg).fit()

def pr_model(name, res):
    print(f"{name}: F={res.fvalue:.3f}, p={res.f_pvalue:.3f}, "
          f"R2={res.rsquared:.3f}, adjR2={res.rsquared_adj:.3f}")

pr_model("CAPM   ", res_capm)
pr_model("2-factor", res_2f)
pr_model("FF3    ", res_ff3)

# Joint F-tests
ft_2f  = res_2f.f_test("SMB = 0")
ft_ff3 = res_ff3.f_test("SMB = 0, HML = 0")

print(f"2-factor H0: SMB=0        -> F={float(ft_2f.fvalue):.3f}, p={float(ft_2f.pvalue):.3f}")
print(f"FF3 joint H0: SMB=HML=0   -> F={float(ft_ff3.fvalue):.3f}, p={float(ft_ff3.pvalue):.3f}")
print("\nWe can conclude all models are dignificant and that extending the model does add value")

#%%
#Problem 7

import numpy as np
import statsmodels.formula.api as smf

# Rebuild factors INCLUDING RF in decimals
fac_full = ff3[['ym', 'mkt_excess_pct', 'SMB', 'HML', 'RF']].copy()
fac_full['Mkt'] = fac_full['mkt_excess_pct'] / 100.0
fac_full['SMB'] = fac_full['SMB'] / 100.0
fac_full['HML'] = fac_full['HML'] / 100.0
fac_full['RF']  = fac_full['RF']  / 100.0
fac_full = fac_full[['ym', 'Mkt', 'SMB', 'HML', 'RF']]

# Merge with Berkshire monthly
reg_full = bh.merge(fac_full, on='ym', how='inner').dropna().reset_index(drop=True)

# Get FF3 exposures (use prior res_ff3 if it exists; otherwise re-fit)
try:
    beta_m = res_ff3.params['Mkt']
    s      = res_ff3.params['SMB']
    h      = res_ff3.params['HML']
except NameError:
    res_ff3 = smf.ols('bh_excess ~ Mkt + SMB + HML', data=reg_full).fit()
    beta_m = res_ff3.params['Mkt']
    s      = res_ff3.params['SMB']
    h      = res_ff3.params['HML']

# Benchmark (total return), BRK (total), Market (total)
reg_full['r_bmk'] = (1 - beta_m) * reg_full['RF'] + beta_m * reg_full['Mkt'] \
                    + s * reg_full['SMB'] + h * reg_full['HML']
reg_full['r_brk'] = reg_full['RF'] + reg_full['bh_excess']
reg_full['r_mkt'] = reg_full['RF'] + reg_full['Mkt']

# $1 growth → log-wealth
for col in ['r_bmk', 'r_brk', 'r_mkt']:
    reg_full[f'logW_{col}'] = np.log((1 + reg_full[col]).cumprod())

# Plot
plt.figure(figsize=(9,5))
plt.plot(reg_full['logW_r_brk'], label='BRK')
plt.plot(reg_full['logW_r_bmk'], label='Benchmark')
plt.plot(reg_full['logW_r_mkt'], label='Market')
plt.title('Log Wealth of $1'); plt.ylabel('log($)')
plt.legend(); plt.tight_layout(); plt.show()
print("\n\nWhat we notice about pre and post 2000 is that pre 2000, BHK outprefomred FF3. Its alpha of 1.81%/month was three times that of FF3. Post 2000, alpha remained positive but was behaving more closer to the FF3 and benchmark. ")


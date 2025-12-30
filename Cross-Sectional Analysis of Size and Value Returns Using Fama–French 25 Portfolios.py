# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 10:38:40 2025

@author: max psihos

"""

#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
    
#2
df = pd.read_csv('25_Portfolios_5x5.csv', skiprows=15, skipinitialspace=True)
#df=df.drop(columns=['Unnamed: 0'])
#Define range of the dataframe using iloc (integer locate)
df1 = df.iloc[:1188]
#Make all values numeric in data frame
df2 = df1.apply(pd.to_numeric, errors="coerce")
#Filter dates for appropriate dates
df3 = df2[df2["Date"] >= 196307]
# Convert YYYYMM integers (e.g., 196307) into datetime
df3["Date"] = pd.to_datetime(df3["Date"].astype(str), format="%Y%m")
#drop NAs from the lists
small_hiBM = df3["SMALL HiBM"].dropna()
big_loBM   = df3["BIG LoBM"].dropna()

#Plot the figure
plt.figure(figsize=(12,6))
plt.plot(df3["Date"], small_hiBM, label="Small-Cap Value (Small HiBM)")
plt.plot(df3["Date"], big_loBM,   label="Large-Cap Growth (Big LoBM)")
#Label the figure
plt.title("Monthly Returns: Small-Cap Value vs. Large-Cap Growth")
plt.xlabel("Date (YYYYMM)")
plt.ylabel("Return (%)")
plt.legend()
plt.grid(True)
plt.show()

#%%
# only select numeric columns
df_num = df3.drop(columns=["Date"])
#finding mean, varaiance, covariance of dataframe3
average_values = df_num.mean()
V = df_num.var()
coV = df_num.cov()

#Checkers
#print(average_values)
#print(V)
#print(coV)

#%%
#Number 3
import matplotlib.cm as cm
import matplotlib.colors as colors
#finding mean, varaiance, covariance of dataframe3
average_values = df_num.mean()
V = df_num.var()
coV = df_num.cov()

# Re-define mean_values here so later code still works
mean_values = average_values.values.reshape(5,5)

# Put the three stats into a dictionary
stats_dict = {"Mean Return": average_values,"Variance": V,"Covariance (vs " + df_num.columns[0] + ")": coV[df_num.columns[0]]}

# Loop over each and plot
for title, values in stats_dict.items():
    # Reshape into 5x5
    grid_values = values.values.reshape(5,5)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    _x = np.arange(5)
    _y = np.arange(5)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    z = np.zeros_like(x)
    dx = dy = 0.5
    dz = grid_values.T.ravel()   # transpose so Size × Value matches

    # Colors
    norm = colors.Normalize(vmin=dz.min(), vmax=dz.max())
    cmap = cm.get_cmap("viridis")   
    colors_values = cmap(norm(dz))

    ax.bar3d(x, y, z, dx, dy, dz, color=colors_values, shade=True)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(["LoBM","BM2","BM3","BM4","HiBM"])
    ax.set_yticklabels(["Small","ME2","ME3","ME4","Big"])
    ax.set_xlabel("Value")
    ax.set_ylabel("Size")
    ax.set_zlabel(title)
    ax.set_title(f"{title} of 25 Portfolios")

    plt.show()


plt.show()

# Lowest and highest mean returns from the average lists above
lowest_return = average_values.idxmin(), average_values.min()
highest_return = average_values.idxmax(), average_values.max()

print("\nLowest return portfolio:", lowest_return)
print("\nHighest return portfolio:", highest_return)

# Mask the diagonal to ignore variance (Had ChatGPT write this and explain it to me to get the maximum of the covariance) 
# I tried to take a values.max of CoV but it was giving incorrect value, this is way above my paygrade but looks cool
coV_no_diag = coV.where(~np.eye(coV.shape[0],dtype=bool))
max_cov_idx = np.unravel_index(np.nanargmax(coV_no_diag.values), coV_no_diag.shape)

portfolio_i = coV.index[max_cov_idx[0]]
portfolio_j = coV.columns[max_cov_idx[1]]
max_cov_value = coV.iloc[max_cov_idx]

print("\nMost strongly covarying portfolios:", portfolio_i, "and", portfolio_j, "with covariance", max_cov_value)

# Variance of each portfolio
highest_variance = V.idxmax(), V.max()
print("\nHighest variance portfolio:", highest_variance)

#%%
# Number/Question 4

# Select both columns together and drop NAs
returns = df3[["Date", "SMALL HiBM", "BIG LoBM"]].dropna().set_index("Date")

# Rolling 12-month compounded returns
ann_returns = returns.rolling(12).apply(lambda x: np.prod(1 + x/100) - 1)

# Drop initial NA rows (first 11 months)
ann_returns = ann_returns.dropna()

# Plot both on the same axis
plt.figure(figsize=(12,6))
plt.plot(ann_returns.index, ann_returns["SMALL HiBM"], label="Small-Cap Value (annualized)")
plt.plot(ann_returns.index, ann_returns["BIG LoBM"], label="Large-Cap Growth (annualized)")
plt.title("Annual Compounded Returns (Rolling 12 Months)")
plt.xlabel("Date")
plt.ylabel("Annual Return")
plt.legend()
plt.grid(True)
plt.show()
#%%
#Number/Question 5
# Keep only the 25 portfolios (drop Date)
returns_all = df3.drop(columns=["Date"])

# Apply rolling compounding column by column
ann_returns_all = returns_all.apply(
    lambda col: col.rolling(12).apply(lambda x: np.prod(1 + x/100) - 1, raw=True)
)

# Drop first 11 months
ann_returns_all = ann_returns_all.dropna()

# Now take the mean annual return per portfolio
annual_means = ann_returns_all.mean()

#Set up figure just like problem 3
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

_x = np.arange(5)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()
z = np.zeros_like(x)
dx = dy = 0.5
dz = mean_values.T.ravel()   # transpose so Size x Value matches

#Set colors
norm = colors.Normalize(vmin=dz.min(), vmax=dz.max())
cmap = cm.get_cmap("viridis")   
colors_values = cmap(norm(dz))

ax.bar3d(x, y, z, dx, dy, dz, color=colors_values, shade=True)

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(["LoBM","BM2","BM3","BM4","HiBM"])
ax.set_yticklabels(["Small","ME2","ME3","ME4","Big"])
ax.set_xlabel("Value")
ax.set_ylabel("Size")
ax.set_zlabel("Mean Annual Return")
ax.set_title("Mean Annual Compounded Returns of 25 Portfolios")

plt.show()
#%%
#Problem/Question 6 

# Select both portfolios using pandas dataframes
returns = df3[["Date", "SMALL HiBM", "BIG LoBM"]].dropna().set_index("Date")

# Rolling 36-month compounded returns (edit same fomrula from question 4 over longer time span)
roll_3yr = returns.rolling(36).apply(lambda x: np.prod(1 + x/100) - 1, raw=True)

# Drop first 35 months
roll_3yr = roll_3yr.dropna()

# Outperformance series equation A-B
roll_3yr["Outperformance"] = roll_3yr["SMALL HiBM"] - roll_3yr["BIG LoBM"]

# Plot both panels in same figure using Matplotlib
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10), sharex=True)

# Top: Small vs Big 3-year returns
ax1.plot(roll_3yr.index, roll_3yr["SMALL HiBM"], label="Small-Cap Value (3yr)")
ax1.plot(roll_3yr.index, roll_3yr["BIG LoBM"], label="Large-Cap Growth (3yr)")
ax1.set_title("3-Year Compounded Returns")
ax1.set_ylabel("Return")
ax1.legend()
ax1.grid(True)

# Bottom: Outperformance
ax2.plot(roll_3yr.index, roll_3yr["Outperformance"], label="Outperformance (Small - Big)", color="purple")
ax2.axhline(0, color="black", linestyle="--")
ax2.set_title("Outperformance of Small-Cap Value vs Large-Cap Growth")
ax2.set_xlabel("Date")
ax2.set_ylabel("Difference in Return")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print values during recessions (set recession years in a list)
recession_years = [1973, 1993, 2000, 2008, 2020]
#Create a for loop using the values provided in the list above
for yr in recession_years:
    # nearest available date
    val = roll_3yr.loc[str(yr)].mean()
    print(f"{yr} Small HiBM: {roll_3yr.loc[str(yr), 'SMALL HiBM'].mean():.3f}, "
          f"Big LoBM: {roll_3yr.loc[str(yr), 'BIG LoBM'].mean():.3f}, "
          f"Outperformance: {roll_3yr.loc[str(yr), 'Outperformance'].mean():.3f}")
#Response to question 6 
print("\n\nAnswer to Question 7:We can see the realtionship between Small HiBM and Big LoBM during these recessions, and that the trend corresponds to small stocks underperforming in recessions which makes sense because if the market is underperforming were going to see the riskier stocks underperform as well.".format())
#%%
#Question/Problem 7
# Number/Question 7

# Monthly returns (already in df3, as %)
monthly_returns = df3["SMALL HiBM"].dropna() / 100  # convert to decimal

# Annual returns (already computed in Q4 -> ann_returns)
annual_returns = ann_returns["SMALL HiBM"].dropna()

# 3-year returns (already computed in Q6 -> roll_3yr)
three_year_returns = roll_3yr["SMALL HiBM"].dropna()

# Plot histograms
fig, axes = plt.subplots(3, 1, figsize=(10,12))

# Monthly
axes[0].hist(monthly_returns, bins=10, density=True, color="steelblue", alpha=0.7)
axes[0].set_title("Small-Cap Value (Monthly Returns)")
axes[0].set_xlabel("Return")
axes[0].set_ylabel("Frequency")

# Annual
axes[1].hist(annual_returns, bins=10, density=True, color="seagreen", alpha=0.7)
axes[1].set_title("Small-Cap Value (Annual Compounded Returns)")
axes[1].set_xlabel("Return")
axes[1].set_ylabel("Frequency")

# 3-Year
axes[2].hist(three_year_returns, bins=10, density=True, color="darkorange", alpha=0.7)
axes[2].set_title("Small-Cap Value (3-Year Compounded Returns)")
axes[2].set_xlabel("Return")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
print("Answer to Question 7: We can see that over time the distribution of the small stocks moves away from a normal distribution and becomes more tailed in both directions. This is because we know that small stocks are more risky, so the risk and reward vary more over time then we see with large cap stocks")
print("\nThis skewness in the distribution as we increase time makes sense then, because there is more time for many different skewing outcomes to take place.")
#%%
#Question/Problem 8 
# Select the two return series (monthly, in decimal form)
#Question/Problem 8 
# Select the two return series (monthly, in decimal form)
small_val = df3["SMALL HiBM"].dropna() / 100
big_growth = df3["BIG LoBM"].dropna() / 100

# Build a 2D histogram using built in np function
counts, xedges, yedges = np.histogram2d(small_val, big_growth, bins=10)

# Create grid
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Set the dimensions of eahc of the bars. 
dx = dy = (xedges[1] - xedges[0]) * 0.9  # bar width ~ bin width
dz = counts.ravel()

# ChatGPT suggetion to get rid of bins with 0 to the data looks more appealing. 
mask = dz > 0
xpos = xpos[mask]
ypos = ypos[mask]
zpos = zpos[mask]
dz   = dz[mask]

# Plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color="skyblue")

ax.set_xlabel("Small Value Return")
ax.set_ylabel("Big Growth Return")
ax.set_zlabel("Count")
ax.set_title("Joint Distribution of Small-Cap Value vs Large-Cap Growth Returns")

plt.show()
print("Answer to Question 8: No, they do not seem to hedge each other.")
print("\n We see that when the big stock performs well, we can see the small stock peforming well, and this makes sense given a positive stock market chagne. We also see the flip, where small and big stocks do poorly at the same time, hence following the market in general")
#%%
# Question/Problem 9

from scipy.stats import skew, kurtosis

# Standardize each portfolio: (x - mean)/std
df_std = (df_num - df_num.mean()) / df_num.std()

# Calculate stats
stats = pd.DataFrame({"Mean": df_std.mean(),"Variance": df_std.var(),"Skewness": df_std.apply(skew),"Excess Kurtosis": df_std.apply(kurtosis, fisher=True)})

print(stats)

# Compare to a standard normal distribution
print("\nReference values for a Standard Normal distribution:")
print("Mean = 0, Variance = 1, Skewness = 0, Excess Kurtosis = 0")
#Explination of problem
print("\nAnswer for Number 9: In comparison, we then can see that the mean is following the general avearge, anything between -1 and 1 and with what looks like a steady trend upward as we climb in stock size.", 
      "\n\nSecondly for Variance, we can see that the variance for this set of data is 'normal'.",
      "\n\nSkewness we see all negative numbers which means our normal distribution is shifted to the left, which could indicate more risk.",
      "\n\nLastly, Kurtosis is higher then a normal distribution and that maeks sense considering our three year time span and the risk of our small stocks. ")
#%%
#Question / Problem 10

from scipy.stats import norm

# Standard normal PDF on the interval of (-4,4) 
x = np.linspace(-4, 4, 200)
pdf = norm.pdf(x, 0, 1)

#make the monthly returns standard (in decimal form) by their types
small_monthly = (small_hiBM/100 - (small_hiBM/100).mean()) / (small_hiBM/100).std()
big_monthly   = (big_loBM/100   - (big_loBM/100).mean())   / (big_loBM/100).std()

#make the annual returns standard as well (same concept as above just using pandas DF)
small_annual = (ann_returns["SMALL HiBM"] - ann_returns["SMALL HiBM"].mean()) / ann_returns["SMALL HiBM"].std()
big_annual   = (ann_returns["BIG LoBM"]   - ann_returns["BIG LoBM"].mean())   / ann_returns["BIG LoBM"].std()

#create the plot
fig, axes = plt.subplots(2, 2, figsize=(12,8))

# Top-left: SM
axes[0,0].hist(small_monthly, bins=120, density=True, alpha=0.6, color="steelblue", label="Histogram")
axes[0,0].plot(x, pdf, "r-", lw=2, label="Normal PDF")
axes[0,0].set_title("Small Value - Monthly Returns")
axes[0,0].set_xlabel("Standardized Return")
axes[0,0].set_ylabel("Density")
axes[0,0].legend()

# Top-right: SA
axes[0,1].hist(small_annual.dropna(), bins=10, density=True, alpha=0.6, color="seagreen", label="Histogram")
axes[0,1].plot(x, pdf, "r-", lw=2, label="Normal PDF")
axes[0,1].set_title("Small Value - Annual Returns")
axes[0,1].set_xlabel("Standardized Return")
axes[0,1].set_ylabel("Density")
axes[0,1].legend()

# Bottom-left: BM
axes[1,0].hist(big_monthly, bins=120, density=True, alpha=0.6, color="darkorange", label="Histogram")
axes[1,0].plot(x, pdf, "r-", lw=2, label="Normal PDF")
axes[1,0].set_title("Big Growth - Monthly Returns")
axes[1,0].set_xlabel("Standardized Return")
axes[1,0].set_ylabel("Density")
axes[1,0].legend()

# Bottom-right: BA
axes[1,1].hist(big_annual.dropna(), bins=10, density=True, alpha=0.6, color="purple", label="Histogram")
axes[1,1].plot(x, pdf, "r-", lw=2, label="Normal PDF")
axes[1,1].set_title("Big Growth - Annual Returns")
axes[1,1].set_xlabel("Standardized Return")
axes[1,1].set_ylabel("Density")
axes[1,1].legend()

plt.tight_layout()
plt.show()

print("\nAnswer to Question 10: The monthly histograms show much fatter tails on either end than the normal curve. Annual returns look closer to normal distribution, but still show heavy tails.")
#%%
#Question / Problem 11
from scipy.stats import probplot

# 2x2 QQ grid
fig, axes = plt.subplots(2, 2, figsize=(12,8))

# Top-left: SM growth values
probplot(small_monthly.dropna(), dist="norm", plot=axes[0,0])
axes[0,0].set_title("QQ Plot - Small Value (Monthly)")

# Top-right: SA growth values
probplot(small_annual.dropna(), dist="norm", plot=axes[0,1])
axes[0,1].set_title("QQ Plot - Small Value (Annual)")

# Bottom-left: BM growth values
probplot(big_monthly.dropna(), dist="norm", plot=axes[1,0])
axes[1,0].set_title("QQ Plot - Big Growth (Monthly)")

# Bottom-right: BA Growth values
probplot(big_annual.dropna(), dist="norm", plot=axes[1,1])
axes[1,1].set_title("QQ Plot - Big Growth (Annual)")

plt.tight_layout()
plt.show()
print("Answer to question 10: The QQ plots show deviations from the expected in the monthly returns",
      "\n\nFor annual returns the QQ follows the normal distribution more closely",
      "\n\nSo, In conclusion financial returns are not normally distributed, but annual returns are closer to normal.")

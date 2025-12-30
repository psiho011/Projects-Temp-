# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:12:21 2025

@author: max psihos
"""

#%%
#Problem 1
print("Problem 1 Answer: They are seperated by 'tabs' and are formatted txt format")
#%%
#Problem 2
import pandas as pd

FF3 = pd.read_csv('FF3factors.txt', sep ='\t', names = ['Year','Month', 'Market Excess Return', 'Small Minus Big','High Minus Large','Risk Free Rate'])
#Using loop logics we can assign the name of either return, yield, or mautrity and give it a correlating number assocaition through the range of 7.
#Added the underscores to the names because APPARENTLY python has an attitude problem
colnames = ['Year', 'Month'] + [f'Return_{i+1}' for i in range(7)] + [f'Yield_{i+1}' for i in range(7)] + [f'Maturity_{i+1}' for i in range(7)]
#Now we can read the txt file and assign the columns the approriate names    
fd = pd.read_csv('USTreturns.txt', sep ='\t', names = colnames)

UST = fd.dropna()
print(UST)

print("Answer for Number 2: All date was in percentage format, nothing needed to be changed for these questions")
#%%
#Problem 3
# Problem 3 filter by supplied dates on the information, years 1963 to 2019, can be adjusted for any data in the data range.
UST1 = UST[((UST['Year'] > 1963) | ((UST['Year'] == 1963) & (UST['Month'] >= 7))) &((UST['Year'] < 2019) | ((UST['Year'] == 2019) & (UST['Month'] <= 12)))]
# Problem 3 stats
maturity_cols = [f"Maturity_{i+1}" for i in range(7)]

# Calculate averages and std deviations using built in pandas funtions
avg_maturities = UST1[maturity_cols].mean()
std_maturities = UST1[maturity_cols].std()

# Combine stats into one big DataFrame, using specifics for orginal maturity, can be altered to grab any years here 
results = pd.DataFrame({'Average': avg_maturities,'StdDev': std_maturities,'Original_Maturity': [1, 2, 3, 5, 7, 10, 20]  })

print(results)
print("short term bonds have small stdev, while long term have larger stdev, both due with their time to maturity")
print("\nLong term bonds are the most varaiable because of the time associated to maturity, short term bonds are the most stable")
#%%
#Problem 4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# first we are going to get the maturities and yields over the range
maturity_cols = [f"Maturity_{i+1}" for i in range(7)]
yield_cols = [f"Yield_{i+1}" for i in range(7)]
#use numpy and dataframes to set up mesh's
M = UST1[maturity_cols].to_numpy()   # maturities (T x 7)
Y = UST1[yield_cols].to_numpy()      # yields (T x 7)

#Set up the X-Axis for time
years = UST1['Year'] + (UST1['Month']-1)/12

#build out meshgrid to baseline maturities 
X, Ymat = np.meshgrid(years, M[0])  
# account for the row by row varaiance 
Ymat = M.T
X = np.tile(years, (M.shape[1], 1))  # repeat time across maturities
Z = UST1[yield_cols].to_numpy().T

# plot the mesh onto the 3d graph
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
#set up the surface of the 3d graph
surf = ax.plot_surface(X, Ymat, Z, cmap=cm.viridis,
                       linewidth=0, antialiased=True)
#label axis' and title of the graphs
ax.set_xlabel('Year')
ax.set_ylabel('Maturity (years)')
ax.set_zlabel('YTM (%)')
ax.set_title('Yield Curve Surfboard (Filled Surface)')
#change colorway of graph to look nicer.
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Yield (%)')
plt.tight_layout()
plt.show()
print("Since the early 1980s, overall interest rates have steadily declined from very high levels to near zero by 2020. Before recessions the yield curve typically flattened or inverted as short-term rates rose, and after recessions it steepened as short-term rates fell more sharply than long-term rates.")
                                     
#%%
#Problem 5

# Using same filter as above, just replaceing UST with FF3
FF3f = FF3[((FF3['Year'] > 1963) | ((FF3['Year'] == 1963) & (FF3['Month'] >= 7))) &
           ((FF3['Year'] < 2019) | ((FF3['Year'] == 2019) & (FF3['Month'] <= 12)))]

#convert the data all to numeric
FF3f['Market Excess Return'] = pd.to_numeric(FF3f['Market Excess Return'], errors='coerce')
#Find the annual mean of the new data we pulled using the filter 
mean_mkt = FF3f['Market Excess Return'].mean() * 12      
#Next same logic, we use bult in functions to find the stdev of the data we are pulling out of the filters
std_mkt = FF3f['Market Excess Return'].std() * np.sqrt(12)  
#then we take the mean over the stdev to find the ratio 
sharpe_mkt = mean_mkt / std_mkt

print("Market Factor:")
print("Annualized Mean Excess Return:", mean_mkt)
print("Annualized Std Dev:", std_mkt)
print("Sharpe Ratio:", sharpe_mkt)

#same logic as above we are just using variables from above to calcuate the same process. 
bond_cols = [f"Return_{i+1}" for i in range(7)]
#use built in functions to find mean/stdev and then the ratio 
bond_means = UST1[bond_cols].mean() * 12
bond_stds = UST1[bond_cols].std() * np.sqrt(12)
bond_sharpes = bond_means / bond_stds

bond_results = pd.DataFrame({'Mean': bond_means,'StdDev': bond_stds,'Sharpe': bond_sharpes,'Original_Maturity': [1, 2, 3, 5, 7, 10, 20]})

print("\nBond Results:")
print(bond_results)

# Finding maximum from the bond restult and then printing it.
best_idx = bond_results['Sharpe'].idxmax()
print("\nThe bond with the highest Sharpe Ratio is", 
      bond_results.loc[best_idx, 'Original_Maturity'], "years to maturity.")
#%%
#problem 6
#Use out dataframes to push together data using pandas concat feature. then filter it down
corr_data = pd.concat([FF3f['Market Excess Return'].reset_index(drop=True), UST1[[f"Return_{i+1}" for i in range(7)]].reset_index(drop=True)], axis=1)
corr_data.columns = ['Market_Excess'] + [f"Return_{i+1}" for i in range(7)]
corr_matrix = corr_data.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Correlation of each bond with the market, using loop to grab all bonds and compare. numpy functions help with the correlation matrix building.
market_corr = corr_matrix.loc['Market_Excess', [f"Return_{i+1}" for i in range(7)]]
print("\nCorrelation of each bond with Market Excess Return:")
print(market_corr)

# Find which bonds are most correlated with each other by using built in numpy functions. 
#again here using a count controlled loop, to get each iteration of the bonds.
bond_corr = corr_matrix.loc[[f"Return_{i+1}" for i in range(7)], [f"Return_{i+1}" for i in range(7)]]
max_corr = (bond_corr.where(~np.eye(bond_corr.shape[0], dtype=bool)).stack().idxmax())
print("\nBonds with highest correlation with each other:", max_corr)
#%%
# Problem 7

# Pick variables: market and 30-year bond using data frames
#need to use reset index here to get back to 0
market = FF3f['Market Excess Return'].reset_index(drop=True)
bond30 = UST1['Return_7'].reset_index(drop=True)

# allign each of the lengths so they are the same size 
n = min(len(market), len(bond30))
market = market[:n]
bond30 = bond30[:n]

# Create 5 bins for each while getting rid of any possible duplicates
market_bins = pd.qcut(market, 5, duplicates='drop')
bond_bins = pd.qcut(bond30, 5, duplicates='drop')

# Cross-tab function
joint_counts = pd.crosstab(market_bins, bond_bins)

print("Joint counts (5x5):")
print(joint_counts)

# Convert to frequency table by dividing counts by sum of all counts
joint_probs = joint_counts / joint_counts.to_numpy().sum()
print("\nJoint frequency table:")
print(joint_probs)

# sum of the rows
marginal_market = joint_probs.sum(axis=1)

# Conditional distribution
cond_probs = joint_probs.div(marginal_market, axis=0)
print("\nConditional distribution P(Bond | Market):")
print(cond_probs)

# Conditional expectation of bond returns given market bucket using mean of the 30yr bond
cond_exp = bond30.groupby(market_bins).mean()
print("\nConditional expectation of 30-year bond given market bucket:")
print(cond_exp)
print("\n When markets are in a bull state, the 30 year bond typically dosent keep up, but when stocks are bearish bonds typically rally. This has to do with market confidence. People want to put their money where its safe when the markets going to shit")
#%%
#Problem 8
import statsmodels.api as sm

# set our variables up to the variables weve found before, market and 30 year bond
y = bond30
x = market
x = sm.add_constant(x)  # add intercept
#build model
model = sm.OLS(y, x).fit()
#set the model parameters
a, b = model.params
print("Regression coefficients: a =", a, " b =", b)
#regression line on -.3 and .3
#made the line here much larger so i could see the intercept more clearly
xmin, xmax = market.min(), market.max()
x_vals = np.linspace(xmin, xmax, 200)
y_vals = a + b * x_vals

# Conditional expectations to use the midpoints of each bin
cond_exp_midpoints = cond_exp.index.map(lambda iv: (iv.left + iv.right)/2)

# Plot the figure, add the data and label the graph
plt.figure(figsize=(8,5))
plt.plot(x_vals, y_vals, label="Regression Line", color="red")
plt.plot(cond_exp_midpoints, cond_exp.values, 'o-', label="Conditional Expectations")

plt.xlabel("Market Excess Return")
plt.ylabel("30-Year Bond Return")
plt.title("30-Year Bond vs Market Excess Return")
plt.legend()
plt.show()
print("/nThe red line is just the best straight-line fit showing how the 30-year bond moves with the market. The blue dots bounce around it, but the line gives a simple summary of the overall trend.")
#%%
#Problem 9

import numpy as np
import matplotlib.pyplot as plt
#set up our x and y variables equal to 30 year bond and market
x = market.values
y = bond30.values

# set the means of each of of the variables above
x_bar = np.mean(x)
y_bar = np.mean(y)

# solve for the slope and intercept, label them as a and b instead of x and y to fit formula
b = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar)**2)
a = y_bar - b * x_bar

# Here is our predicition, and then the associated r^2 calculation.
y_hat = a + b * x
sst = np.sum((y - y_bar)**2)
sse = np.sum((y - y_hat)**2)
r2 = 1 - sse/sst

print("Manual regression: a =", a, " b =", b)
print("R^2 =", r2)

# Scatter plot with regression line
plt.figure(figsize=(8,5))
plt.scatter(x, y, alpha=0.4, label="Data points")
plt.plot(np.sort(x), a + b * np.sort(x), color="red", label="Regression Line")
plt.xlabel("Market Excess Return")
plt.ylabel("30-Year Bond Return")
plt.title("30-Year Bond vs Market Excess Return (Manual Regression)")
plt.legend()
plt.show()

print("The market return does a pretty bad job of explaining the 30-year bond, the line is almost flat and the R² is tiny. The line from this regression basically matches the one we saw before, but it’s smoother since it uses every data point instead of grouped averages, while the conditional expectations bounce around more because they’re based on bins.")





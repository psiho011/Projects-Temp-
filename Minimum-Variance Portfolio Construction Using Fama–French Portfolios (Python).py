# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:26:01 2025

@author: mpsih
"""

#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
    
df = pd.read_csv('25_Portfolios_5x5.csv', skiprows=15, skipinitialspace=True)
#df=df.drop(columns=['Unnamed: 0'])
#Define range of the dataframe using iloc (integer locate)
df1 = df.iloc[:1188, [0,5,10,15,20]]
#turn all strings to numerical values so we can sum later
df1 = df1.apply(pd.to_numeric)

#converting percent to decimals 
df2=df1/100
#finding mean of dataframe2
average_values = df2.mean()
print(average_values)
#%%
#Converting into a numpy array 
# assume df2 = cleaned DataFrame (N×5), already scaled to decimals from division above
lol = df2.to_numpy()              # shape (N,5)
XD = lol - lol.mean(axis=0)         # center each column (N,5)
N = lol.shape[0] 

Sigma = (XD.T @ XD) / (N - 1) 
#Assign values to the new data frame
Sigma_df = pd.DataFrame(Sigma, index=df2.columns, columns=df2.columns)
print(Sigma_df)
#%%
#First define the shape
n = Sigma.shape[0]
#Setting up our matrix here
A_top = np.hstack([2*Sigma, np.ones((n,1))])
A_bottom = np.hstack([np.ones((1,n)), np.zeros((1,1))])
#Here we are using numpy to stack the variables and build the matrix 
A = np.vstack([A_top, A_bottom])
#Building right hand ride of the matrix with 1s
b = np.vstack([np.zeros((n,1)), np.ones((1,1))])
#solving for the linear equation 
x = np.linalg.inv(A) @ b
#turn into single dimensional array 
weights = x[:n].flatten()
#Give the return over the time series at given weights 
r_p = df2.to_numpy() @ weights   # (N,5) @ (5,) = (N,)


#compound to get total return over the whole time series
R_total = np.prod(1 + r_p) - 1

print("Monthly portfolio returns (first 5):", r_p[:5])
print("Total portfolio return:", R_total)
#8 Answer
#we want to rebalance the portfolio at the end of each month because if we dont we could see the spread start to grow between the secrutities.
#For example, if one of the stocks outweights the others and continues to outperform and we dont rebalance, the portfolio risk may start to rise.
#%%
starting_value=100 
#Had to look up this function, added cummulative to the function in order to get each value across the time series 
values = starting_value*np.cumprod(1+r_p)
#Auditing my answers (blanking out but leaving in for me to have)
#print("First 5 r_p:", r_p[:5])
#print("First 5 values:", values[:5])
#print("Final value:", values[-1])
plt.figure()
plt.plot(values, label="Portfolio Values", linewidth=2) #Plotting the values of the lines
#name the graph and axis and legend
plt.title("Growth of $100 in a Weighted portflio")
plt.xlabel("Time (months)")
plt.ylabel("Value ($)" )
plt.legend()
plt.grid(True, linestyle = "--")
plt.show()
#%%
#Here just use numpy to do natty log of the values
log_values = np.log(values)

plt.figure()
plt.plot(log_values, label="ln(Portfolio Values)", linewidth=2)
#Again just naming the graph and axis' the same as above and assigning legend
plt.title("Natural Log of Portfolio Value Over Time")
plt.xlabel("Time (months)")
plt.ylabel("ln(Value) ($)")
plt.legend()
plt.grid(True, linestyle="--")
plt.show()


#%%



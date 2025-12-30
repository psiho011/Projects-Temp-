# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:26:01 2025

@author: mpsih
"""

#%%
import pandas as pd 
import numpy as np 
    
df = pd.read_csv('25_Portfolios_5x5.csv', skiprows=15, skipinitialspace=True)
df=df.drop(columns=['Unnamed: 0'])
#bm1_cols = [c for c in df.columns if ('BM1' in c) or ('LoBM' in c)]
#df[bm1_cols] = df[bm1_cols].apply(pd.to_numeric, errors='coerce')/100

df1 = df.iloc[:1187, [0,5,10,15,20]]
df1 = df1.apply(pd.to_numeric)
#df["Rohil"]= pd.to_numeric(df["Rohil"])
#converting percent to decimals 
df2=df1/100
#finding mean of dataframe2
average_values = df2.mean()
print(average_values)
#Converting into a numpy array 
# assume df2 = cleaned DataFrame (N×5), already scaled to decimals
X = df2.to_numpy()              # shape (N,5)
Xc = X - X.mean(axis=0)         # center each column (N,5)
N = X.shape[0] 

Sigma = (Xc.T @ Xc) / (N - 1) 
Sigma_df = pd.DataFrame(Sigma, index=df2.columns, columns=df2.columns)
print(Sigma_df)

n = Sigma.shape[0]
A_top = np.hstack([2*Sigma, np.ones((n,1))])
A_bottom = np.hstack([np.ones((1,n)), np.zeros((1,1))])
A = np.vstack([A_top, A_bottom])
b = np.vstack([np.zeros((n,1)), np.ones((1,1))])
x = np.linalg.inv(A) @ b
weights = x[:n].flatten()
r_p = df2.to_numpy() @ weights   # (N,5) @ (5,) = (N,)


# 2. compound to get total return over the whole period
R_total = np.prod(1 + r_p) - 1

print("Monthly portfolio returns (first 5):", r_p[:5])
print("Total portfolio return:", R_total)

#%%



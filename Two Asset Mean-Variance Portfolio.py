# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
###Excersise 1
x = input("Input a number:")
x1=float(x)
fun = 4*(x1**3)-3*(x1**2)-2*(x1)+1
print(fun)
#%%
###Excersise 2 
x = input("Input a number:")
x1=float(x)

lol = 12*(x1**3)-6*(x1**2)-2
print(lol)

#%%
#Excersise 3
import numpy as np

xp = np.arange(-1.5,1.5+0.001,0.001)
slopes = 12 * (xp**2)-(6*xp)-2

results = np.column_stack((xp,slopes))
print(results)
#%%
#Excersise 4
import matplotlib.pyplot as plt 
import numpy as np

xp = np.arange(-1.5,1.5+0.001,0.001) 
f_x=4*(xp**3)-3*(xp**2)-2**xp+1 #orginal function
slopes = 12 * (xp**2)-(6*xp)-2 #Derivative of funtion

plt.figure()
plt.plot(xp,f_x,label="f(x) = 4x^3 - 3x^2 - 2x + 1", color = "b")
plt.plot(xp,slopes,label="f'(x) = 12x^2 - 6x - 2", color = "r")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
#%%
#Excersise 5 
import math 

def f(x): return 4*x**3-3*x**2-2*x+x1
def ff(x): return 12*x**2-6*x-2
def fff(x): return 24*x-6

a = 12
b = -6
c = -2

slopey=b*b-4*a*c
x1= (-b - math.sqrt(slopey))/(2*a)
x2= (-b + math.sqrt(slopey))/(2*a)

for x in (x1, x2):
    if fff(x)>0:
        kind="local minimum"
    else:
        kind ="local maximum"
    print(f"{kind} at x = {x:.5f}, f(x) = {f(x):.5f}")
#%%
#Excercise 6 
import matplotlib.pyplot as plt
import math

s1 = 0.3
s2 = 0.4
rho = 0.5

s1_2 = s1**2
s2_2 = s2**2
r12 = rho*s1*s2

# Derivative of variance: a*w + b
a = 2*(s1_2 + s2_2 - 2*r12)
b = -2*s2_2 + 2*r12

w_star = -b/a

var_min = s1_2*w_star**2 + s2_2*(1-w_star)**2 + 2*w_star*(1-w_star)*r12

w = np.linspace(0, 1, 500)
var_p = s1_2*w**2 + s2_2*(1-w)**2 + 2*w*(1-w)*r12

plt.plot(w, var_p, label="Portfolio Variance", color="g")
plt.scatter(w_star, var_min, color="r", zorder=5, label="Minimum")
plt.xlabel("Weight in Asset 1 (w)")
plt.ylabel("Variance")
plt.title("Minimum Variance Portfolio")
plt.legend()
plt.grid(True)
plt.show()
print(f"Minimum variance is at w = {w_star:.5f}, Var = {var_min:.5f}")

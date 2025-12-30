import pandas as pd
import matplotlib.pyplot as plt

# ---- Inputs ----
P = float(input("Enter loan amount ($): "))
annual_rate = float(input("Enter annual interest rate (%): ")) / 100
years = int(input("Enter loan term (years): "))

# ---- Payment math ----
r = annual_rate / 12
n = years * 12

if r == 0:
    payment = P / n
else:
    payment = P * (r * (1 + r)**n) / ((1 + r)**n - 1)

print(f"Monthly payment: ${payment:,.2f}")

# ---- Amortization schedule ----
balance = P
rows = []

for m in range(1, n + 1):
    interest = balance * r
    principal = payment - interest

    # Guard for final rounding so balance doesn't go negative
    if principal > balance:
        principal = balance
        payment_actual = principal + interest
    else:
        payment_actual = payment

    balance -= principal

    rows.append({
        "Month": m,
        "Payment": round(payment_actual, 2),
        "Principal": round(principal, 2),
        "Interest": round(interest, 2),
        "Balance": round(balance, 2)
    })

df = pd.DataFrame(rows)

display(df.head(12))      # first year
display(df.tail(12))      # last year

print(f"Total interest paid: ${df['Interest'].sum():,.2f}")
print(f"Total paid: ${df['Payment'].sum():,.2f}")

# ---- Plots ----
plt.figure()
plt.plot(df["Month"], df["Balance"])
plt.xlabel("Month")
plt.ylabel("Remaining balance ($)")
plt.title("Loan balance over time")
plt.show()

plt.figure()
plt.plot(df["Month"], df["Interest"], label="Interest")
plt.plot(df["Month"], df["Principal"], label="Principal")
plt.xlabel("Month")
plt.ylabel("Monthly amount ($)")
plt.title("Monthly payment split: interest vs principal")
plt.legend()
plt.show()

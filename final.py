import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from risk3 import get_stock_data, train_data, test_data

# -------------------------------
# Tickers
# -------------------------------
tickers = ["SPY.US", "TLT.US", "GLD.US", "XLF.US", "XLK.US"]
# -------------------------------
# Parameters
# -------------------------------
train_start = dt.datetime(2010, 1, 1)
train_end   = dt.datetime(2020, 1, 1)
test_start  = dt.datetime(2021, 1, 1)
test_end    = dt.datetime(2023, 12, 31)   # safe end date for Stooq

# -------------------------------
# Train + Test
# -------------------------------
results = {}

for t in tickers:
    try:
        # Training
        X_train, _, _ = get_stock_data(t, train_start, train_end)
        calm, stress, model, mu, sigma = train_data(X_train)

        # Testing
        cp, sp, r_test, _ = test_data(model, calm, stress, mu, sigma, t, test_start, test_end)

        # Store
        results[t] = {
            "returns": r_test,
            "stress_prob": pd.Series(sp, index=r_test.index)
        }

        print(f"{t} loaded: {len(r_test)} test points")

    except Exception as e:
        print(f"{t} skipped: {e}")

# -------------------------------
# Build returns & stress DataFrames
# Outer join ensures no empty portfolio
# -------------------------------
returns_df = pd.DataFrame({t: results[t]["returns"] for t in results})
stress_df = pd.DataFrame({t: results[t]["stress_prob"] for t in results})

# Missing data: zero returns, full stress
returns_df = returns_df.fillna(0.0)
stress_df = stress_df.fillna(1.0)

# -------------------------------
# Dynamic portfolio weights
# -------------------------------
raw_weights = 1.0 - stress_df
weights = raw_weights.div(raw_weights.sum(axis=1), axis=0).fillna(0.0)

# -------------------------------
# Portfolio & benchmark
# -------------------------------
portfolio_returns = (weights * returns_df).sum(axis=1)
portfolio_value = (1.0 + portfolio_returns).cumprod()

benchmark_returns = returns_df.mean(axis=1)
benchmark_value = (1.0 + benchmark_returns).cumprod()

# -------------------------------
# Plots
# -------------------------------
plt.figure(figsize=(12,6))
portfolio_value.plot(label="GMM-HMM Regime Portfolio")
benchmark_value.plot(label="Equal-weight Benchmark", linestyle="--")
plt.title("Portfolio Value (2021–2023)")
plt.ylabel("Cumulative Value")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
weights.plot(legend=False)
plt.title("Dynamic Portfolio Weights")
plt.ylabel("Weight")
plt.xlabel("Date")
plt.grid(True)
plt.show()

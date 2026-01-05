from hmmlearn.hmm import GMMHMM
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as pdr

# -------------------------------
# Load data from Stooq
# -------------------------------
def get_stock_data(ticker, start, end):
    # Load
    data = pdr.DataReader(ticker, "stooq", start, end)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Ensure datetime index
    if not np.issubdtype(data.index.dtype, np.datetime64):
        data.index = pd.to_datetime(data.index)

    data = data.sort_index()

    # Stooq columns
    if "close" in data.columns:
        prices = data["close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        raise KeyError(f"No Close column found for {ticker}. Columns: {data.columns}")

    # Returns
    returns = prices.pct_change().dropna()
    X = returns.values.reshape(-1, 1)

    return X, returns, prices.loc[returns.index]

# -------------------------------
# Train GMM-HMM
# -------------------------------
def train_data(X_train):
    model = GMMHMM(
        n_components=2,      # Calm / Stress
        n_mix=2,             # 2 Gaussians per regime
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    mu = X_train.mean()
    sigma = X_train.std()
    X_train_norm = (X_train - mu) / sigma

    model.fit(X_train_norm)

    # Identify calm vs stress by volatility
    state_vol = model.covars_.reshape(2, -1).mean(axis=1)
    calm_state = np.argmin(state_vol)
    stress_state = np.argmax(state_vol)

    return calm_state, stress_state, model, mu, sigma

# -------------------------------
# Test / get regime probabilities
# -------------------------------
def test_data(model, calm_state, stress_state, mu, sigma, ticker, start, end):
    X_test, returns_test, prices_test = get_stock_data(ticker, start, end)

    # Standardize using training parameters
    X_test_norm = (X_test - mu) / sigma

    # Predict probabilities
    state_probs = model.predict_proba(X_test_norm)

    calm_prob = state_probs[:, calm_state]
    stress_prob = state_probs[:, stress_state]

    return calm_prob, stress_prob, returns_test, prices_test
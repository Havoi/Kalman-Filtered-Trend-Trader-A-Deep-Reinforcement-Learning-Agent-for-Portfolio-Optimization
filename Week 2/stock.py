import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# --- 1. Get Real Data ---
# We'll download Apple (AAPL) data for the last year
ticker = "AAPL"
data = yf.download(ticker, period="1y", interval="1d")
prices = data['Close'].values

# Handle potentially different data structures from yfinance
if prices.ndim > 1:
    prices = prices.flatten()

# --- 2. Initialize Kalman Filter Variables ---
# We use the exact matrix logic from Module 2

# State Vector [Price, Trend]
# Initial guess: Price = First real price, Trend = 0
x = np.array([[prices[0]], 
              [0]])

# Covariance Matrix P (Uncertainty)
P = np.eye(2) * 1000

# State Transition Matrix F (Physics)
# Next Price = Current Price + Current Trend
# Next Trend = Current Trend
F = np.array([[1, 1], 
              [0, 1]])

# Measurement Matrix H (Observer)
# We only see Price (1, 0)
H = np.array([[1, 0]])

# --- CRITICAL: TUNING FOR FINANCE ---

# R: Measurement Noise (Market Volatility)
# High R = Filter ignores daily spikes (Smooth). Low R = Filter chases price.
# We set this to the variance of the first 30 days to adapt to the stock.
R = np.array([[np.var(prices[:30])]]) 

# Q: Process Noise (How fast the "Truth" changes)
# Q[0,0] (Price noise): Small wiggle
# Q[1,1] (Trend noise): VERY IMPORTANT. 
# If this is high, the filter reacts to trend changes instantly (jittery).
# If this is low, the filter requires a lot of evidence to change the trend (smooth).
Q = np.array([[0.001, 0], 
              [0, 0.001]]) 

# Storage for plotting
estimated_prices = []
estimated_trends = []

# --- 3. The Loop (Simulating Real-Time Trading) ---
for price in prices:
    
    # A. PREDICT
    # Project state ahead
    x = np.dot(F, x)
    # Project uncertainty
    P = np.dot(np.dot(F, P), F.T) + Q
    
    # B. UPDATE
    z = np.array([[price]]) # The current market price
    
    # Innovation (Error)
    y = z - np.dot(H, x)
    
    # System Uncertainty (S)
    S = np.dot(np.dot(H, P), H.T) + R
    
    # Kalman Gain (K)
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    
    # Update State
    x = x + np.dot(K, y)
    
    # Update Covariance
    I = np.eye(2)
    P = np.dot((I - np.dot(K, H)), P)
    
    # Store results
    estimated_prices.append(x[0, 0])
    estimated_trends.append(x[1, 0])

# --- 4. Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top Plot: Price vs Estimate
ax1.plot(prices, 'k.', alpha=0.3, label='Market Price (Noisy)')
ax1.plot(estimated_prices, 'b-', linewidth=2, label='Kalman Estimate')
ax1.set_title(f'Kalman Filter on {ticker} - Price Smoother')
ax1.set_ylabel('Price ($)')
ax1.legend()
ax1.grid(True)

# Bottom Plot: The Hidden Trend (Velocity)
# This is what we trade on!
zeros = np.zeros(len(estimated_trends))
ax2.plot(estimated_trends, 'r-', linewidth=1.5, label='Estimated Trend (Slope)')
ax2.fill_between(range(len(estimated_trends)), estimated_trends, zeros, 
                 where=(np.array(estimated_trends) > 0), color='green', alpha=0.3, label='Buy Zone')
ax2.fill_between(range(len(estimated_trends)), estimated_trends, zeros, 
                 where=(np.array(estimated_trends) < 0), color='red', alpha=0.3, label='Sell Zone')
ax2.axhline(0, color='black', linewidth=1)
ax2.set_title('The Signal: Estimated Trend Velocity')
ax2.set_ylabel('Daily Price Change ($/day)')
ax2.set_xlabel('Days')
ax2.legend()
ax2.grid(True)

plt.show()
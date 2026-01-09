import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Generate Synthetic Pairs Data ---
np.random.seed(42)
n_steps = 300
time = np.linspace(0, 10, n_steps)

# Stock A (The Independent Mover)
price_A = np.sin(time) * 10 + 100 + np.random.normal(0, 1, n_steps)

# Stock B (The Dependent Mover)
# Intention: Relies on A, but the relationship changes halfway through!
# First half: B = 1.5 * A + 10
# Second half: B = 0.5 * A + 10 (Regime Change)
true_beta = np.concatenate([np.ones(150)*1.5, np.ones(150)*0.5])
price_B = true_beta * price_A + 10 + np.random.normal(0, 1, n_steps)

# --- 2. Kalman Filter for Regression ---

# State: [Beta, Alpha]
# Initial Guess: Beta=1, Alpha=0
state = np.array([[1.0], 
                  [0.0]])

# Covariance P
P = np.eye(2) * 10.0

# Process Noise Q (Random Walk for Beta and Alpha)
# We allow Beta to drift slightly over time
Q = np.array([[1e-5, 0], 
              [0, 1e-5]])

# Measurement Noise R
R = np.array([[2.0]]) # Variance of the noise in the spread

beta_estimates = []
alpha_estimates = []

# --- 3. The Loop ---
for t in range(n_steps):
    
    # Observe the prices
    # We are trying to predict B using A
    # y = beta * x + alpha
    # So: Measurement = Price_B, Input = Price_A
    x_input = price_A[t]
    y_measurement = price_B[t]
    
    # A. PREDICT
    # State stays same (Random Walk), but uncertainty grows
    # F is Identity because parameters shouldn't change without data
    state = state # x_new = x_old
    P = P + Q
    
    # B. UPDATE
    # H changes every step! It contains the price of Stock A
    H = np.array([[x_input, 1.0]])
    
    # Measurement y is Price_B
    z = np.array([[y_measurement]])
    
    # Innovation: Error in prediction
    y_innov = z - np.dot(H, state)
    
    # S = H P H.t + R
    S = np.dot(np.dot(H, P), H.T) + R
    
    # Kalman Gain
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    
    # Update State
    state = state + np.dot(K, y_innov)
    
    # Update P
    I = np.eye(2)
    P = np.dot((I - np.dot(K, H)), P)
    
    # Store
    beta_estimates.append(state[0, 0])
    alpha_estimates.append(state[1, 0])

# --- 4. The "Spread" Calculation ---
# The Spread is the difference between Actual Price and Model Price
# Spread = Price_B - (Beta * Price_A + Alpha)
spread = price_B - (np.array(beta_estimates) * price_A + np.array(alpha_estimates))

# --- 5. Visualization ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: The Prices
ax1.plot(price_A, label='Stock A (Driver)')
ax1.plot(price_B, label='Stock B (Follower)')
ax1.set_title('The Asset Pair')
ax1.legend()
ax1.grid(True)

# Plot 2: The Estimated Beta (Hidden State)
ax2.plot(true_beta, 'g--', label='True Beta (Hidden Regime)')
ax2.plot(beta_estimates, 'b-', linewidth=2, label='Kalman Estimated Beta')
ax2.set_ylabel('Hedge Ratio')
ax2.set_title('Dynamic Beta Estimation')
ax2.legend()
ax2.grid(True)

# Plot 3: The Trading Signal (Spread)
# If Spread is roughly 0, the model fits.
# If Spread spikes, it's a trading opportunity (Mean Reversion)
ax3.plot(spread, 'purple', label='Spread (Error)')
ax3.axhline(0, color='k', linestyle='--')
ax3.set_title('The Spread (Trading Signal)')
ax3.set_ylabel('Deviation')
ax3.legend()
ax3.grid(True)

plt.show()
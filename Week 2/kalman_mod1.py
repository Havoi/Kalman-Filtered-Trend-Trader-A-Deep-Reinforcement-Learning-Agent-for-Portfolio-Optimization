import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate Synthetic Data ---
np.random.seed(42)
n_steps = 100
time = np.linspace(0, 10, n_steps)

# The "True" Value (Hidden from the trader)
true_price = np.sin(time) * 10 + 100  # Oscillating between 90 and 110

# The "Measurement" (What you see on the screen)
# R represents market microstructure noise (bid-ask bounce, panic)
R_true_sim = 2.0  
measurement_noise = np.random.normal(0, R_true_sim, n_steps)
noisy_price = true_price + measurement_noise

# --- 2. The Kalman Filter Setup ---

# Initial Guesses
estimate = 100.0   # x: Initial estimate of price
uncertainty = 5  # P: Initial uncertainty (variance)

# Hyperparameters (The "Knobs" you turn)
Q = 0.1   # Process Noise: How much we think the true value changes naturally
R = 2.0 # Measurement Noise: How much we trust the market price (Variance)

estimates = []
kalman_gains = []

# --- 3. The Loop (Predict -> Update) ---
for measure in noisy_price:
    
    # A. PREDICT STEP
    # In a simple 1D model, we predict the price stays the same, 
    # but uncertainty grows because time has passed.
    estimate_pred = estimate
    uncertainty_pred = uncertainty + Q
    
    # B. UPDATE STEP
    # Calculate Kalman Gain (K)
    # K = Error in Estimate / (Error in Estimate + Error in Measurement)
    K = uncertainty_pred / (uncertainty_pred + R)
    
    # Update estimate based on the measurement (measure)
    # New = Pred + Gain * (Observed - Pred)
    estimate = estimate_pred + K * (measure - estimate_pred)
    
    # Update Uncertainty (P)
    # We are more certain now, so variance shrinks
    uncertainty = (1 - K) * uncertainty_pred
    
    # Store data
    estimates.append(estimate)
    kalman_gains.append(K)

# --- 4. Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(time, true_price, 'g--', label='True Value (Hidden)', alpha=0.7)
plt.plot(time, noisy_price, 'k.', label='Noisy Measurements (Market Price)', alpha=0.3)
plt.plot(time, estimates, 'b-', label='Kalman Filter Estimate', linewidth=2)
plt.title('Module 1: 1D Kalman Filter - Signal Extraction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Let's check the convergence of K
# In a static system, K usually settles to a constant value
print(f"Final Kalman Gain: {kalman_gains[-1]:.4f}")
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup
# We are VERY sure about Price (0.1), but unsure about Trend (5.0)
P = np.array([[0.1, 0], 
              [0, 5.0]])

# Physics: Price moves, Trend stays constant
F = np.array([[1, 1], 
              [0, 1]])

# Process Noise (Fog added every step)
Q = np.array([[0.1, 0], 
              [0, 0.1]])

print("--- Step 0: Initial Uncertainty ---")
print(P)

# 2. PREDICT STEP (Eyes Closed)
# The blob should grow and stretch
P_pred = np.dot(np.dot(F, P), F.T) + Q

print("\n--- Step 1: After Prediction (Eyes Closed) ---")
print("Notice the Top-Left (Price Variance) increased massively!")
print("Because uncertainty in Trend 'leaked' into Price.")
print(P_pred)

# 3. UPDATE STEP (Eyes Open)
# We see a measurement with low noise (R=1)
R = np.array([[1.0]])
H = np.array([[1, 0]]) # We measure Price

# Calculate Kalman Gain (Simplified for demo)
# K ~ P / (P + R)
S = np.dot(np.dot(H, P_pred), H.T) + R
K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))

# Shrink the uncertainty
I = np.eye(2)
P_updated = np.dot((I - np.dot(K, H)), P_pred)

print("\n--- Step 2: After Update (Eyes Open) ---")
print("Notice the numbers shrank. We are confident again.")
print(P_updated)
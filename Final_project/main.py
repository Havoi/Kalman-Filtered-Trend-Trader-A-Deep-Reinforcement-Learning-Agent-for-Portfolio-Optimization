import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces

# ==========================================
# PART 1: KALMAN FILTER IMPLEMENTATION
# ==========================================
class KalmanFilterTrend:
    """
    A 1D Kalman Filter to estimate Position (Price) and Velocity (Trend).
    State Vector x = [Price, Trend]^T
    """
    def __init__(self, R=1.0, Q=0.1):
        self.dt = 1.0  # Time step
        
        # State Transition Matrix (F)
        # x_new = x_old + v_old * dt
        # v_new = v_old
        self.F = np.array([[1, self.dt],
                           [0, 1]])
        
        # Measurement Matrix (H)
        # We only observe Price
        self.H = np.array([[1, 0]])
        
        # Measurement Noise Covariance (R)
        self.R = np.array([[R]])
        
        # Process Noise Covariance (Q) - Tunable
        self.Q = np.array([[Q/10, 0],
                           [0, Q]])
        
        # Initial State and Covariance
        self.x = np.zeros((2, 1)) # [Price, Trend]
        self.P = np.eye(2) * 100

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        # z is the noisy measurement (Close Price)
        y = z - self.H @ self.x # Residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.x = self.x + K @ y
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x

# ==========================================
# PART 2: DATA PIPELINE
# ==========================================
def get_data(tickers, start_date, end_date):
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Handle missing values (Forward fill then Backward fill)
    data = data.ffill().bfill()
    
    # Align timestamps is automatic with pandas, but we ensure order
    data = data[tickers] 
    return data

def apply_kalman_filters(df, R_val=10, Q_val=0.01):
    trends = pd.DataFrame(index=df.index, columns=df.columns)
    smoothed = pd.DataFrame(index=df.index, columns=df.columns)
    
    for col in df.columns:
        kf = KalmanFilterTrend(R=R_val, Q=Q_val)
        kf.x[0, 0] = df[col].iloc[0] # Initialize with first price
        
        slope_list = []
        smooth_list = []
        
        for price in df[col].values:
            kf.predict()
            est = kf.update(price)
            smooth_list.append(est[0, 0])
            slope_list.append(est[1, 0])
            
        trends[col] = slope_list
        smoothed[col] = smooth_list
        
    return trends, smoothed

# ==========================================
# PART 3: RL ENVIRONMENT
# ==========================================
class PortfolioEnv(gym.Env):
    def __init__(self, price_df, trend_df, initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        
        self.prices = price_df
        self.trends = trend_df
        self.assets = price_df.columns # ['BTC-USD', 'GC=F', '^NSEI']
        self.n_assets = len(self.assets)
        self.initial_balance = initial_balance
        
        # Transaction cost: 0.1% 
        self.cost_bps = 0.001 
        
        # Action Space: 0=Cash, 1=BTC, 2=Gold, 3=Nifty, 4=Balanced
        self.action_space = spaces.Discrete(self.n_assets + 2)
        
        # State Space: [Trend_1, Trend_2, Trend_3, Vol_1, Vol_2, Vol_3, Current_Weights...]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets * 3 + 1,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None):
        self.current_step = 60 # Start after minimal window for volatility
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.n_assets) # Quantity of each asset
        self.cash = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.prev_value = self.initial_balance
        
        # Track weights for state [Cash, Asset1, Asset2, Asset3]
        self.current_weights = np.array([1.0] + [0.0]*self.n_assets)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Normalized Trends (Slope / Price)
        curr_prices = self.prices.iloc[self.current_step]
        curr_trends = self.trends.iloc[self.current_step]
        norm_trends = (curr_trends / curr_prices).values
        
        # 2. Volatility (20-day rolling std dev of returns)
        pct_change = self.prices.iloc[self.current_step-20 : self.current_step].pct_change()
        volatility = pct_change.std().fillna(0).values
        
        # 3. Current Portfolio Weights (excluding cash for simplicity in observation)
        obs = np.concatenate([norm_trends, volatility, self.current_weights])
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Determine Target Allocation based on Action
        # Actions: 0=Cash, 1=BTC, 2=Gold, 3=Nifty, 4=Balanced
        target_w = np.zeros(self.n_assets)
        
        if action == 0:
            pass # All Cash
        elif action == 4:
            target_w[:] = 1.0 / self.n_assets # Balanced risky
        else:
            target_w[action - 1] = 1.0 # 100% allocation to specific asset
            
        # 2. Execute Rebalancing
        current_prices = self.prices.iloc[self.current_step].values
        current_val = self.cash + np.sum(self.holdings * current_prices)
        
        # Target value for each asset
        # If action is 0 (Cash), target_val_assets is 0
        target_val_assets = current_val * target_w 
        
        # Calculate Rebalancing Cost
        current_val_assets = self.holdings * current_prices
        diff = np.abs(target_val_assets - current_val_assets)
        transaction_cost = np.sum(diff) * self.cost_bps
        
        # Update Holdings
        self.holdings = target_val_assets / current_prices
        self.cash = current_val - np.sum(target_val_assets) - transaction_cost
        
        # 3. Step Time Forward
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        # 4. Calculate New Portfolio Value
        new_prices = self.prices.iloc[self.current_step].values
        new_val = self.cash + np.sum(self.holdings * new_prices)
        self.portfolio_value = new_val
        
        # Update weights for next state
        total = new_val
        w_assets = (self.holdings * new_prices) / total
        w_cash = self.cash / total
        self.current_weights = np.concatenate([[w_cash], w_assets])

        # 5. Reward: Log Return - Cost Penalty
        ret = (new_val - self.prev_value) / self.prev_value
        reward = ret 
        
        self.prev_value = new_val
        
        return self._get_obs(), reward, done, False, {}

# ==========================================
# PART 4: DQN AGENT
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim)
        self.target = DQN(input_dim, output_dim)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.action_size = output_dim

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target(next_state_t)).item()
            
            state_t = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_t)
            target_f[0][action] = target
            
            loss = nn.MSELoss()(self.model(state_t), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

# ==========================================
# PART 5: MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    tickers = ['BTC-USD', 'GC=F', '^NSEI'] # [cite: 10, 11, 12]
    raw_data = get_data(tickers, '2015-01-01', '2024-12-31')
    
    # 2. Kalman Filter Feature Engineering
    trend_data, smooth_data = apply_kalman_filters(raw_data)
    
    # Visual check
    plt.figure(figsize=(10,6))
    plt.plot(raw_data['^NSEI'], label='Nifty Raw', alpha=0.5)
    plt.plot(smooth_data['^NSEI'], label='Nifty Kalman', linestyle='--')
    plt.title("Kalman Filter Tracking Verification")
    plt.legend()
    plt.show()

    # 3. Train/Test Split
    split_idx = int(len(raw_data) * 0.7)
    train_prices = raw_data.iloc[:split_idx]
    train_trends = trend_data.iloc[:split_idx]
    
    test_prices = raw_data.iloc[split_idx:]
    test_trends = trend_data.iloc[split_idx:]
    
    # 4. Initialize Env and Agent
    env = PortfolioEnv(train_prices, train_trends)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent(state_dim, action_dim)
    
    # 5. Training Loop
    episodes = 20
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            agent.replay()
        
        agent.update_target()
        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    # 6. Backtesting (Out of Sample)
    print("\nStarting Backtest...")
    test_env = PortfolioEnv(test_prices, test_trends)
    state, _ = test_env.reset()
    done = False
    
    portfolio_values = []
    agent.epsilon = 0 # No exploration in backtest
    
    while not done:
        action = agent.act(state)
        state, _, done, _, _ = test_env.step(action)
        portfolio_values.append(test_env.portfolio_value)
        
    # 7. Metrics & Benchmark
    results = pd.DataFrame(portfolio_values, columns=['Strategy'])
    results.index = test_prices.index[61:] # Adjust for gym step offset
    
    # Benchmark: Nifty Buy and Hold [cite: 49]
    nifty_bh = test_prices['^NSEI'].iloc[61:]
    results['Benchmark'] = (nifty_bh / nifty_bh.iloc[0]) * 10000
    
    # Calculate Sharpe Ratio
    strat_ret = results['Strategy'].pct_change().dropna()
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    
    print(f"Final Strategy Value: {results['Strategy'].iloc[-1]:.2f}")
    print(f"Final Benchmark Value: {results['Benchmark'].iloc[-1]:.2f}")
    print(f"Strategy Sharpe Ratio: {sharpe:.2f}")
    
    results.plot()
    plt.title("WiDS Project: Kalman-DQN Strategy vs Nifty 50")
    plt.show()
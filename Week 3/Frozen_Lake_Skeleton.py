import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import pickle
from collections import deque

# -----------------------------
# Configuration
# -----------------------------
N_TRIALS = 100         # Increased optimization attempts
N_EPISODES = 5000      # Increased training length per agent
SAVE_FILE = "best_q_table.pkl"

# Global variables to store data for plotting/saving
best_global_score = -1.0
best_Q_table = None
study_results = []  # To store {alpha, gamma, decay, score} for every trial
best_learning_curve = [] # To store the curve of the champion

def train_agent(trial):
    global best_global_score, best_Q_table, best_learning_curve

    # 1. Hyperparameter Search Space
    alpha = trial.suggest_float("alpha", 0.05, 0.9)
    gamma = trial.suggest_float("gamma", 0.8, 0.9999)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.99995)
    
    # Setup
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    epsilon = 1.0
    epsilon_min = 0.01
    
    # Tracking
    success_window = deque(maxlen=100)
    history = [] 
    peak_score_this_trial = 0.0

    # 2. Training Loop
    for episode in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        for step in range(100):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q-Value
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state

            if done:
                # 1.0 for goal, 0.0 for hole/timeout
                success_window.append(reward) 
                break
        
        # Calculate Rolling Mean
        current_mean = np.mean(success_window) if len(success_window) == 100 else 0
        peak_score_this_trial = max(peak_score_this_trial, current_mean)
        
        # Save history for plotting (downsampled to save RAM)
        if episode % 10 == 0:
            history.append(current_mean)
            
        # Pruning: Stop hopeless trials early
        trial.report(current_mean, episode)
        if trial.should_prune():
            raise optuna.TrialPruned()

    env.close()

    # 3. Save "Champion" Artifacts
    if peak_score_this_trial > best_global_score:
        best_global_score = peak_score_this_trial
        best_Q_table = Q.copy()
        best_learning_curve = history
        # Save to disk immediately
        with open(SAVE_FILE, "wb") as f:
            pickle.dump(Q, f)

    # Store data for visualization
    study_results.append({
        "alpha": alpha,
        "gamma": gamma,
        "decay": epsilon_decay,
        "score": peak_score_this_trial
    })

    return peak_score_this_trial

# -----------------------------
# Run Optimization
# -----------------------------
print(f"Starting Optimization ({N_TRIALS} trials, {N_EPISODES} episodes each)...")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Clean output
study = optuna.create_study(direction="maximize")
study.optimize(train_agent, n_trials=N_TRIALS)

print(f"\nOptimization Complete!")
print(f"Best Peak Success Rate: {study.best_value * 100:.2f}%")
print(f"Best Params: {study.best_params}")
print(f"Best Q-table saved to {SAVE_FILE}")

# -----------------------------
# Beautiful Visualizations
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")
fig = plt.figure(figsize=(20, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Extract Data
alphas = [r['alpha'] for r in study_results]
gammas = [r['gamma'] for r in study_results]
decays = [r['decay'] for r in study_results]
scores = [r['score'] for r in study_results]

# GRAPH 1: The "Sweet Spot" Heatmap (Alpha vs Gamma)
# We use a hexbin plot to show where high scores cluster
ax1 = plt.subplot(2, 2, 1)
hb = ax1.hexbin(alphas, gammas, C=scores, gridsize=20, cmap='viridis', reduce_C_function=np.max)
cb = fig.colorbar(hb, ax=ax1)
cb.set_label('Max Success Rate')
ax1.set_xlabel("Learning Rate (Alpha)")
ax1.set_ylabel("Discount Factor (Gamma)")
ax1.set_title("The 'Goldilocks Zone': Alpha vs Gamma")

# GRAPH 2: Parameter Importance (Regression Plots)
# Does higher Decay always mean better score?
ax2 = plt.subplot(2, 2, 2)
sns.regplot(x=decays, y=scores, ax=ax2, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
ax2.set_xlabel("Epsilon Decay")
ax2.set_ylabel("Peak Success Rate")
ax2.set_title("Impact of Decay Rate on Success")

# GRAPH 3: High-Dimensional Parallel Coordinates
# Shows the path of parameters for the top 10 trials
ax3 = plt.subplot(2, 2, 3)
from optuna.visualization.matplotlib import plot_parallel_coordinate
# We generate this using Optuna's internal tool but render to our subplot
# Note: Optuna's plotting functions are complex to embed in subplots directly, 
# so we create a custom simplified parallel plot for top 10 performers.
top_results = sorted(study_results, key=lambda x: x['score'], reverse=True)[:10]
for res in top_results:
    # Normalize values for plotting 0-1 scale relative to search space
    norm_alpha = (res['alpha'] - 0.05) / (0.9 - 0.05)
    norm_gamma = (res['gamma'] - 0.8) / (0.9999 - 0.8)
    norm_decay = (res['decay'] - 0.99) / (0.99995 - 0.99)
    
    ax3.plot(['Alpha', 'Gamma', 'Decay'], [norm_alpha, norm_gamma, norm_decay], 
             marker='o', alpha=0.7, label=f"{res['score']:.2f}")

ax3.set_yticks([])
ax3.set_title("Parameter Paths of Top 10 Agents (Normalized)")
ax3.text(0, -0.1, "Low", ha='center', transform=ax3.transAxes)
ax3.text(0, 1.05, "High", ha='center', transform=ax3.transAxes)

# GRAPH 4: The Champion's Learning Curve
ax4 = plt.subplot(2, 2, 4)
ax4.plot(best_learning_curve, color="#2ecc71", linewidth=2.5)
ax4.fill_between(range(len(best_learning_curve)), best_learning_curve, color="#2ecc71", alpha=0.1)
ax4.set_xlabel("Training Episodes (x10)")
ax4.set_ylabel("Success Rate (Rolling 100)")
ax4.set_title(f"Champion Agent Learning Curve (Score: {best_global_score:.2f})")
ax4.set_ylim(0, 1.05)

plt.show()
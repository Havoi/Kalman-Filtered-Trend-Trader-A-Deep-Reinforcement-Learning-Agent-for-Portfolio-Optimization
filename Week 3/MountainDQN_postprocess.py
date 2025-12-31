import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import random
from collections import deque
import matplotlib.pyplot as plt

# ==========================================
# 1. HELPER FUNCTIONS & CONFIG
# ==========================================

def normalize_state(state):
    """
    Scales raw state features to [-1, 1].
    """
    pos = state[0]
    vel = state[1]
    
    # Position: [-1.2, 0.6] -> [-1, 1]
    norm_pos = (pos - (-1.2)) / (0.6 - (-1.2)) 
    norm_pos = norm_pos * 2 - 1 
    
    # Velocity: [-0.07, 0.07] -> [-1, 1]
    norm_vel = (vel - (-0.07)) / (0.07 - (-0.07))
    norm_vel = norm_vel * 2 - 1
    
    return np.array([norm_pos, norm_vel])

# ==========================================
# 2. CORE CLASSES (Memory & Neural Net)
# ==========================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # He Initialization
        scale = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.biases = np.zeros((1, output_dim))
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values, learning_rate):
        d_weights = np.dot(self.inputs.T, d_values)
        d_biases = np.sum(d_values, axis=0, keepdims=True)
        d_inputs = np.dot(d_values, self.weights.T)

        # Gradient Clipping
        d_weights = np.clip(d_weights, -1, 1)
        d_biases = np.clip(d_biases, -1, 1)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_inputs

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_values):
        d_inputs = d_values.copy()
        d_inputs[self.inputs <= 0] = 0
        return d_inputs

class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [
            DenseLayer(input_size, hidden_size),
            ReLU(),
            DenseLayer(hidden_size, hidden_size),
            ReLU(),
            DenseLayer(hidden_size, output_size)
        ]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_values, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                d_values = layer.backward(d_values, learning_rate)
            else:
                d_values = layer.backward(d_values)

    def get_weights(self):
        return [(l.weights.copy(), l.biases.copy()) for l in self.layers if isinstance(l, DenseLayer)]

    def set_weights(self, weights):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.weights = weights[idx][0].copy()
                layer.biases = weights[idx][1].copy()
                idx += 1

# ==========================================
# 3. THE AGENT (Modified for Logging)
# ==========================================

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.learning_rate = 0.001
        self.batch_size = 64

        self.main_network = SimpleNeuralNet(state_size, 64, action_size)
        self.target_network = SimpleNeuralNet(state_size, 64, action_size)
        self.memory = ReplayBuffer(20000)
        self.update_target_network()

    def update_target_network(self):
        weights = self.main_network.get_weights()
        self.target_network.set_weights(weights)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = state.reshape(1, -1)
        q_values = self.main_network.forward(state)
        return np.argmax(q_values)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0 # No loss yet

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.amax(next_q_values, axis=1)
        targets = rewards + (self.gamma * max_next_q * (1 - dones))

        current_q_values = self.main_network.forward(states)
        
        # Calculate Loss (Mean Squared Error) for logging
        # We only care about the error of the action taken
        rows = np.arange(self.batch_size)
        predicted = current_q_values[rows, actions]
        error = predicted - targets
        loss = np.mean(np.square(error)) # Log this!

        # Backprop
        d_loss = np.zeros_like(current_q_values)
        d_loss[rows, actions] = error
        d_loss /= self.batch_size 
        self.main_network.backward(d_loss, self.learning_rate)
        
        return loss

    def save_model(self, filename='dqn_model.npz'):
        """
        Saves the Main Network weights to a compressed .npz file.
        """
        params = {}
        # Iterate through layers and grab weights/biases
        for i, layer in enumerate(self.main_network.layers):
            if isinstance(layer, DenseLayer):
                params[f'w_{i}'] = layer.weights
                params[f'b_{i}'] = layer.biases
        
        # Save to disk
        np.savez_compressed(filename, **params)
        print(f"Model saved to {filename}")

    def load_model(self, filename='dqn_model.npz'):
        """
        Loads weights from a .npz file into the Main Network.
        """
        data = np.load(filename)
        
        # Iterate through layers and inject weights/biases
        for i, layer in enumerate(self.main_network.layers):
            if isinstance(layer, DenseLayer):
                if f'w_{i}' in data:
                    layer.weights = data[f'w_{i}']
                    layer.biases = data[f'b_{i}']
        
        # Sync target network so it matches the loaded brain
        self.update_target_network()
        print(f"Model loaded from {filename}")
        
        
        
# ==========================================
# 4. TRAINING WITH ANALYTICS
# ==========================================

if __name__ == "__main__":
    env = gym.make('MountainCar-v0', render_mode=None)
    env._max_episode_steps = 1000 # Give it time to win
    
    agent = DQNAgent(state_size=2, action_size=3)
    
    # ANALYTICS STORAGE
    history = {
        'rewards': [],
        'avg_rewards': [],
        'loss': [],
        'epsilon': [],
        'max_height': []
    }
    
    episodes = 400
    epsilon_decay = 0.95
    
    print("--- Starting Training (Data Science Mode) ---")

    for e in range(episodes):
        raw_state, _ = env.reset()
        state = normalize_state(raw_state)
        
        total_reward = 0
        episode_loss = []
        max_pos = -1.2
        done = False
        
        while not done:
            action = agent.act(state)
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = normalize_state(next_raw_state)
            
            # Analytics: Track how high we got
            if next_raw_state[0] > max_pos:
                max_pos = next_raw_state[0]
            
            # Reward Shaping
            pos = next_raw_state[0]
            vel = next_raw_state[1]
            height_reward = abs(pos - (-1.2)) 
            kinetic_reward = abs(vel) * 10 
            modified_reward = height_reward + kinetic_reward
            
            if pos >= 0.5:
                modified_reward += 50.0
            
            agent.memory.push(state, action, modified_reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Train and track loss
            loss = agent.train()
            if loss != 0:
                episode_loss.append(loss)

        # End of Episode Updates
        agent.update_target_network() # Simplified: Update every episode for stability
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= epsilon_decay
            
        # --- LOGGING DATA ---
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        history['rewards'].append(total_reward)
        history['loss'].append(avg_loss)
        history['epsilon'].append(agent.epsilon)
        history['max_height'].append(max_pos)
        
        # Calculate Moving Average (last 50 episodes)
        avg_rew = np.mean(history['rewards'][-50:])
        history['avg_rewards'].append(avg_rew)
        
        print(f"Ep: {e+1} | Score: {total_reward:.1f} | MaxHeight: {max_pos:.2f} | Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f}")

        # Stop if strictly solved
        if avg_rew > -130 and e > 100:
            print("Solved! Agent is consistently reaching the top.")
            agent.save_model("mountaincar_solved.npz")
            break

    env.close()

    # ==========================================
    # 5. POST-PROCESSING: THE DASHBOARD
    # ==========================================
    print("\nGenerating Analytics Dashboard...")
    plt.style.use('ggplot') # Make it pretty
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'DQN Training Analysis (From Scratch) - {len(history["rewards"])} Episodes', fontsize=16)

    # Plot 1: Reward History
    axs[0, 0].plot(history['rewards'], alpha=0.3, color='blue', label='Raw Reward')
    axs[0, 0].plot(history['avg_rewards'], color='darkblue', linewidth=2, label='Moving Avg (50)')
    axs[0, 0].set_title('Reward Consistency')
    axs[0, 0].set_ylabel('Total Score')
    axs[0, 0].legend()

    # Plot 2: Loss (The "Brain" Metric)
    axs[0, 1].plot(history['loss'], color='red', alpha=0.6)
    axs[0, 1].set_title('Training Loss (MSE)')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_yscale('log') # Log scale is better for loss

    # Plot 3: Max Height (The "Goal" Metric)
    axs[1, 0].plot(history['max_height'], color='green', alpha=0.6)
    axs[1, 0].axhline(y=0.5, color='black', linestyle='--', label='Flag (Goal)')
    axs[1, 0].set_title('Max Height Reached per Episode')
    axs[1, 0].legend()

    # Plot 4: Epsilon Decay
    axs[1, 1].plot(history['epsilon'], color='purple')
    axs[1, 1].set_title('Exploration Rate (Epsilon)')
    
    plt.tight_layout()
    plt.savefig('dqn_analytics_dashboard.png')
    plt.show()
    print("Dashboard saved as 'dqn_analytics_dashboard.png'")

    # ==========================================
    # 6. VIDEO GENERATION
    # ==========================================
    print("\nRecording Victory Lap Video...")
    
    # We create a specific folder for the video
    # episode_trigger=lambda x: True records every episode (we only run 1)
    video_env = gym.make('MountainCar-v0', render_mode='rgb_array')
    video_env = RecordVideo(video_env, video_folder='./mountaincar_video', episode_trigger=lambda x: True)

    raw_state, _ = video_env.reset()
    state = normalize_state(raw_state)
    done = False
    
    while not done:
        # No random actions, pure exploitation
        state = state.reshape(1, -1)
        action = np.argmax(agent.main_network.forward(state))
        
        next_raw_state, _, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated
        state = normalize_state(next_raw_state)
        
    video_env.close()
    print("Video saved in './mountaincar_video' folder!")
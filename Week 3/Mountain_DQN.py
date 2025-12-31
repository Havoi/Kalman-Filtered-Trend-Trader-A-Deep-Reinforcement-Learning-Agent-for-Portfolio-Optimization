import numpy as np
import gymnasium as gym
import random
from collections import deque
import math

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def normalize_state(state):
    """
    Scales the raw state features to a range of roughly [-1, 1].
    Why? Neural networks learn much faster when inputs are small and centered 
    around zero. Raw velocity values are too small (0.07) compared to position.
    """
    # MountainCar limits: Position [-1.2, 0.6], Velocity [-0.07, 0.07]
    pos = state[0]
    vel = state[1]
    
    # Scale Position to approx [-1, 1]
    norm_pos = (pos - (-1.2)) / (0.6 - (-1.2)) 
    norm_pos = norm_pos * 2 - 1 
    
    # Scale Velocity to approx [-1, 1]
    norm_vel = (vel - (-0.07)) / (0.07 - (-0.07))
    norm_vel = norm_vel * 2 - 1
    
    return np.array([norm_pos, norm_vel])

# ==========================================
# 2. THE MEMORY (REPLAY BUFFER)
# ==========================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of experiences.
        # This breaks correlations (e.g., sequential frames are too similar).
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. THE NEURAL NETWORK (FROM SCRATCH)
# ==========================================

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # HE INITIALIZATION (Crucial Fix):
        # We scale weights by sqrt(2/n). This keeps the signal magnitude 
        # consistent when passing through ReLU layers.
        scale = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.biases = np.zeros((1, output_dim))

        self.inputs = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, inputs):
        self.inputs = inputs
        # Y = X . W + B
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values, learning_rate):
        # 1. Gradient for Weights: X^T . dY
        self.d_weights = np.dot(self.inputs.T, d_values)
        
        # 2. Gradient for Biases: Sum of dY (col-wise)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        
        # 3. Gradient for Inputs (to pass back): dY . W^T
        d_inputs = np.dot(d_values, self.weights.T)

        # 4. Update Parameters (Gradient Descent)
        # We clip gradients slightly to prevent math overflows in raw numpy
        self.d_weights = np.clip(self.d_weights, -1, 1)
        self.d_biases = np.clip(self.d_biases, -1, 1)

        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

        return d_inputs

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_values):
        d_inputs = d_values.copy()
        # Derivative of ReLU is 1 if x > 0, else 0
        d_inputs[self.inputs <= 0] = 0
        return d_inputs

class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Architecture: Input -> Dense -> ReLU -> Dense -> ReLU -> Output
        self.layers = [
            DenseLayer(input_size, hidden_size),
            ReLU(),
            DenseLayer(hidden_size, hidden_size),
            ReLU(),
            DenseLayer(hidden_size, output_size) # Output is Linear (Raw Q-Values)
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
# 4. THE DQN AGENT
# ==========================================

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.05     # Keep 5% exploration always
        self.epsilon_decay = 0.99   # Slow decay to ensure it finds the goal
        self.learning_rate = 0.0005 # Slower learning rate for stability
        self.batch_size = 64

        # Networks
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
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 1. Get Target Q-values
        # "What does the Target Network think is the best move from the NEXT state?"
        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.amax(next_q_values, axis=1)

        # 2. Bellman Equation
        # Target = Reward + Gamma * Max(Next_Q)
        targets = rewards + (self.gamma * max_next_q * (1 - dones))

        # 3. Get Current Q-values
        current_q_values = self.main_network.forward(states)

        # 4. Compute Gradient (Error)
        # We only want to update the Q-value for the action we ACTUALLY took.
        # Error = Current_Q - Target
        d_loss = np.zeros_like(current_q_values)
        rows = np.arange(self.batch_size)
        
        # d_loss for the specific actions taken
        d_loss[rows, actions] = (current_q_values[rows, actions] - targets)
        
        # Optional: Divide by batch size effectively scales the gradient 
        # to be the average error, making learning rate independent of batch size.
        d_loss /= self.batch_size 

        # 5. Backpropagate
        self.main_network.backward(d_loss, self.learning_rate)

    

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    # Create Environment
    env = gym.make('MountainCar-v0', render_mode=None)
    
    # FIX 1: Increase the time limit so the car has time to swing up
    # The default is 200. We change it to 1000.
    env._max_episode_steps = 1000 
    
    agent = DQNAgent(state_size=2, action_size=3)

    episodes = 500
    target_update_freq = 10
    
    # We will decay epsilon manually in the loop now, not inside the agent
    epsilon_decay = 0.95 # Decay by 5% every episode

    print("Starting Training...")
    
    for e in range(episodes):
        raw_state, _ = env.reset()
        state = normalize_state(raw_state)
        
        total_reward = 0
        done = False
        steps_taken = 0
        
        while not done:
            action = agent.act(state)
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            
            # Check if done
            done = terminated or truncated
            
            next_state = normalize_state(next_raw_state)
            
            # Reward Shaping
            pos = next_raw_state[0]
            vel = next_raw_state[1]
            
            # Height reward (0 to 1.8)
            height_reward = abs(pos - (-1.2)) 
            # Velocity reward (encourages movement)
            kinetic_reward = abs(vel) * 10 
            
            modified_reward = height_reward + kinetic_reward
            
            if pos >= 0.5:
                modified_reward += 50.0 # Huge bonus for winning
                print(f"!!! GOAL REACHED AT EPISODE {e} (Step {steps_taken}) !!!")
                # We can stop the episode immediately if we want, or let it finish
            
            agent.memory.push(state, action, modified_reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps_taken += 1
            
            agent.train()

        # Update target network
        if e % target_update_freq == 0:
            agent.update_target_network()

        # FIX 2: Decay Epsilon ONCE PER EPISODE
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= epsilon_decay

        print(f"Episode: {e+1}/{episodes}, Steps: {steps_taken}, Real Score: {total_reward:.1f}, Epsilon: {agent.epsilon:.2f}")
        
        if total_reward > -150 and e > 50:
            print("Solved! Stopping training.")
            break

    # Visualization code remains the same...
    print("\nTraining Finished. Watching the agent play...")
    env.close()
    env = gym.make('MountainCar-v0', render_mode='human')
    raw_state, _ = env.reset()
    state = normalize_state(raw_state)
    done = False
    while not done:
        env.render()
        state = state.reshape(1, -1)
        action = np.argmax(agent.main_network.forward(state))
        next_raw_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = normalize_state(next_raw_state)
    env.close()
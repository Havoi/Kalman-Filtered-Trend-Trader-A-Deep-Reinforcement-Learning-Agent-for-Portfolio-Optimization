import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# --- 1. DATA LOADING & PREPROCESSING ---
def load_data():
    print("Loading MNIST data...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Flatten: (60000, 28, 28) -> (784, 60000)
    # Normalize: 0-255 -> 0-1
    X_train = train_images.reshape(train_images.shape[0], -1).T / 255.
    X_test = test_images.reshape(test_images.shape[0], -1).T / 255.
    
    # One-Hot Encode Training Labels
    digits = 10
    Y_train = np.eye(digits)[train_labels].T
    
    return X_train, Y_train, train_labels, X_test, test_labels

# --- 2. MATH FUNCTIONS ---
def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def Softmax(Z):
    # Shift Z for numerical stability (prevents overflow)
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def compute_loss(A2, Y):
    m = Y.shape[1]
    # Add epsilon to prevent log(0)
    log_probs = np.multiply(np.log(A2 + 1e-15), Y)
    cost = - (1/m) * np.sum(log_probs)
    return cost

# --- 3. CORE NEURAL NETWORK FUNCTIONS ---
def init_params():
    # He Initialization
    W1 = np.random.uniform(-0.5,0.5,size = (128,784))
    b1 = np.zeros((128, 1))
    W2 = np.random.uniform(-0.5,0.5,size = (10,128))
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = Softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    
    # Output Layer Gradients
    dZ2 = A2 - Y
    dW2 = (1/m) * (dZ2 @ A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Hidden Layer Gradients
    dZ1 = (W2.T @ dZ2) * deriv_ReLU(Z1)
    dW1 = (1/m) * (dZ1 @ X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# --- 4. MODEL MANAGEMENT (SAVE/LOAD) ---
def save_model(filename, W1, b1, W2, b2):
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Model saved to {filename}")

def load_model(filename):
    data = np.load(filename)
    return data['W1'], data['b1'], data['W2'], data['b2']

# --- 5. TRAINING LOOP WITH VISUALIZATION ---
def gradient_descent(X, Y, true_labels, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    
    # History lists for plotting
    accuracy_history = []
    loss_history = []
    
    print(f"Training on {X.shape[1]} examples for {iterations} iterations...")
    
    for i in range(iterations):
        # Forward & Backward
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Track metrics every 10 iterations
        if i % 10 == 0:
            loss = compute_loss(A2, Y)
            predictions = np.argmax(A2, axis=0)
            acc = np.mean(predictions == true_labels)
            
            loss_history.append(loss)
            accuracy_history.append(acc)
            
            if i % 100 == 0:
                print(f"Iter {i}: Loss {loss:.4f} | Accuracy {acc:.4f}")
                
    return W1, b1, W2, b2, loss_history, accuracy_history

# --- 6. PLOTTING FUNCTION ---
def plot_metrics(loss_hist, acc_hist):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_hist, label='Training Loss', color='red')
    plt.title('Loss over Iterations')
    plt.xlabel('Iterations (x10)')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc_hist, label='Training Accuracy', color='blue')
    plt.title('Accuracy over Iterations')
    plt.xlabel('Iterations (x10)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.show()

# --- 7. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Setup
    X_train, Y_train_enc, Y_train_raw, X_test, Y_test_raw = load_data()
    
    # B. Train
    # Using a higher alpha (0.15) and 800 iterations for better results
    W1, b1, W2, b2, losses, accuracies = gradient_descent(X_train, Y_train_enc, Y_train_raw, 0.05, 2001)
    
    # C. Visualize
    plot_metrics(losses, accuracies)
    
    # D. Save
    save_model('mnist_model_v1.npz', W1, b1, W2, b2)
    
    # E. Test on new data (Proof of Reuse)
    print("\n--- Testing Model Reuse ---")
    W1_loaded, b1_loaded, W2_loaded, b2_loaded = load_model('mnist_model_v1.npz')
    
    # Pick a random test image
    idx = np.random.randint(0, X_test.shape[1])
    current_image = X_test[:, idx, None] # Shape (784, 1)
    current_label = Y_test_raw[idx]
    
    _, _, _, A2_test = forward_prop(W1_loaded, b1_loaded, W2_loaded, b2_loaded, current_image)
    prediction = np.argmax(A2_test)
    
    print(f"Test Image Index: {idx}")
    print(f"True Label: {current_label}")
    print(f"Model Prediction: {prediction}")
    
    # Show the image
    plt.imshow(current_image.reshape(28, 28), cmap='gray')
    plt.title(f"True: {current_label}, Pred: {prediction}")
    plt.show()
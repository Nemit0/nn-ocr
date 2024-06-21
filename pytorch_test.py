import numpy as np

# Define sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define sigmoid derivative
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Loss function (Mean Squared Error)
def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate synthetic data
np.random.seed(0)
X = np.array([0+0.001*i for i in range(100)]).reshape(-1, 1)
y = np.array([np.sin(x) for x in X]).reshape(-1, 1)

# Initialize weights and bias
W = np.random.rand(1, 1)
b = np.random.rand(1)

# Training hyperparameters
learning_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)
    
    # Compute loss
    current_loss = loss(y, y_pred)
    
    # Backward pass
    d_loss = y_pred - y
    d_z = d_loss * sigmoid_derivative(z)
    dW = np.dot(X.T, d_z) / len(X)
    db = np.sum(d_z) / len(X)
    
    # Update weights and bias
    W -= learning_rate * dW
    b -= learning_rate * db
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {current_loss}')

print("Training complete.")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data() #Retrieving the images from MNIST database

train_images = train_images.reshape(-1, 784) / 255.0 #Normalization
test_images  = test_images.reshape(-1, 784) / 255.0

def vectorize(labels, num=10): #Convert numbers into 1x10 vector
    encoded = np.zeros((labels.size, num))
    encoded[np.arange(labels.size), labels] = 1
    return encoded

train_labels_v = vectorize(train_labels)
test_labels_v  = vectorize(test_labels)

#Initializes few parameters
input_nodes  = 784
hidden_nodes = 128
output_nodes = 10

lr = 0.1
epochs = 30
batch_size = 128 

#Get random values for the weight and biases
W1 = np.random.randn(input_nodes, hidden_nodes) / input_nodes
b1 = np.zeros((1, hidden_nodes))

W2 = np.random.randn(hidden_nodes, output_nodes) / input_nodes
b2 = np.zeros((1, output_nodes))

#Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

samples = train_images.shape[0]

for epoch in range(epochs):

    order = np.random.permutation(samples) #Randomize input images
    X = train_images[order]
    Y = train_labels_v[order]

    for i in range(0, samples, batch_size):

        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]

        z1 = x_batch @ W1 + b1 #Forward pass
        a1 = relu(z1)

        z2 = a1 @ W2 + b2
        y_pred = relu(z2)

        dz2 = y_pred - y_batch #Backpropagation
        dW2 = a1.T @ dz2 / batch_size
        db2 = np.mean(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = x_batch.T @ dz1 / batch_size
        db1 = np.mean(dz1, axis=0, keepdims=True)

        W2 -= lr * dW2 #Update
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    #Calculate Losses
    z1 = train_images @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    loss = -np.mean(np.sum(train_labels_v * np.log(relu(z2) + 1e-8), axis=1))

    print(f"Epoch {epoch+1}/{epochs}, Loss = {loss:.4f}")

#Test accuracy
z1 = test_images @ W1 + b1 
a1 = relu(z1)
z2 = a1 @ W2 + b2
predictions = np.argmax(relu(z2), axis=1)

accuracy = np.mean(predictions == test_labels)
print("Test Accuracy:", accuracy)

i = np.random.randint(0, len(test_images))
plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
plt.title(f"Prediction: {predictions[i]} | True: {test_labels[i]}")
plt.axis('off')
plt.show()


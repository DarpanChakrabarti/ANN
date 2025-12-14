import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values (0-255 -> 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images (28x28 -> 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 2. Build the ANN model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')])

# 3. Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
print("\nTraining the model...\n")
model.fit(x_train, y_train, epochs=5)

# 5. Evaluate on test data
print("\nEvaluating on test data...\n")
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# 6. Predict a sample image
index = 500  # Change this to try different images

plt.imshow(x_test[index].reshape(28,28), cmap='gray')
plt.title("Actual Label: " + str(y_test[index]))
plt.show()

prediction = model.predict(x_test[index].reshape(1,784))
predicted_label = np.argmax(prediction)

print("Predicted Digit =", predicted_label)
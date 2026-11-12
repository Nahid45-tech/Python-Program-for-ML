# Neural Network for XOR Problem using TensorFlow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Define dataset (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2. Build the neural network
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 3. Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 4. Train the model for 500 epochs
history = model.fit(X, y, epochs=500, verbose=0)

# 5. Print final loss and accuracy
final_loss = history.history["loss"][-1]
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Final Training Loss: {final_loss:.6f}")
print(f"Training Accuracy: {accuracy:.4f}")

# 6. Show predictions for all inputs
predictions = model.predict(X, verbose=0)
print("\nPredictions:")
for inp, pred in zip(X, predictions):
    print(f"{inp} -> {pred[0]:.4f}")

# 7. Plot training loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], color="blue", linewidth=2)
plt.title("Training Loss Over Epochs (XOR Neural Network)")
plt.xlabel("Epoch")
plt.ylabel("Loss (Binary Crossentropy)")
plt.grid(True)
plt.show()

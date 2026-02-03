import tensorflow as tf 
# TensorFlow is an open-source machine learning library popular in industry
# recent versions of TensorFlow automatically detect if there is a GPU available for computation
import matplotlib 
matplotlib.use('Agg')  # avoids GUI window on macOS terminal
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd() # function that checks where the script is being run from
print(f"The 'plots' directory is being created relative to: {current_dir}")

os.makedirs("plots", exist_ok=True) # Create a folder "plots" to save plots

# Load the Fashion-MNIST dataset 
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

# Normalize pixel values to 0-1
train_images = train_images / 255.0
valid_images = valid_images / 255.0

# Number of classes (as plain Python int)
number_of_classes = int(train_labels.max() + 1)

# Builds the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(number_of_classes)
])

model.summary()

# Compiles the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Visualize one training image
data_idx1 = 42
plt.figure()
plt.imshow(train_images[data_idx1], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.title(f"Label: {train_labels[data_idx1]}")
plt.savefig(f"plots/image_{data_idx1}.png")
plt.close()

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=True,
    validation_data=(valid_images, valid_labels)
)

# Predict on first 10 training images
predictions = model.predict(train_images[0:10])

# Study a single example with bar chart
data_idx2 = 8675
plt.figure()
plt.imshow(train_images[data_idx2], cmap='gray')
plt.title(f"Label: {train_labels[data_idx2]}")
plt.colorbar()
plt.grid(False)
plt.savefig(f"plots/image_{data_idx2}.png")
plt.close()

plt.figure()
x_values = range(number_of_classes)
plt.bar(x_values, predictions[0].flatten())  # example prediction
plt.xticks(range(number_of_classes))
plt.xlabel("Class")
plt.ylabel("Logit value")
plt.title("Model predictions (first example)")
plt.savefig("plots/prediction_bar.png")
plt.close()

print("\nAll plots saved in 'plots/' folder.\n")
print("Example correct label (index data 42):", train_labels[data_idx1])
print("Example correct label (index data 8675):", train_labels[data_idx2])

categories = """
Label | Description
------|------------
  0   | T-shirt/top
  1   | Trouser
  2   | Pullover
  3   | Dress
  4   | Coat
  5   | Sandal
  6   | Shirt
  7   | Sneaker
  8   | Bag
  9   | Ankle boot
"""

print(categories)
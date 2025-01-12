# Import necessary libraries
import tensorflow as tf  # Main deep learning library
from tensorflow import keras  # High-level neural network API
import matplotlib.pyplot as plt  # For plotting graphs
import cv2  # For image processing
import numpy as np  # For numerical operations
from google.colab import files  # For handling file uploads in Google Colab
from IPython.display import Image, display  # For displaying images in notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from sklearn.model_selection import train_test_split  # For splitting dataset

# Load the MNIST dataset which contains 70,000 grayscale images of handwritten digits
print("Loading and training model...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1 to make training more stable
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Split training data into training and validation sets
# 80% for training, 20% for validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# Reshape the data to include a channel dimension (required for CNN input)
# -1 means "figure out this dimension automatically"
# 28x28 is the image size, 1 is the number of color channels (grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Create a data generator that will augment training images
# This helps prevent overfitting by creating variations of the training data
datagen = ImageDataGenerator(
    rotation_range=15,  # Randomly rotate images by up to 15 degrees
    width_shift_range=0.15,  # Randomly shift images horizontally
    height_shift_range=0.15,  # Randomly shift images vertically
    zoom_range=0.15,  # Randomly zoom images
    preprocessing_function=lambda x: 1 - x if np.random.random() > 0.5 else x  # Randomly invert colors
)

# Create the neural network model
model = keras.Sequential([
    # Flatten the 28x28 image into a 1D array
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    # Dense layers with different numbers of neurons
    # ReLU activation function introduces non-linearity
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),  # Normalize the inputs to reduce internal covariate shift
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    # Output layer with 10 neurons (one for each digit)
    # Softmax activation gives probability distribution over all digits
    keras.layers.Dense(10, activation='softmax')
])

# Configure the model for training
model.compile(
    optimizer='adam',  # Adam optimizer automatically adjusts learning rate
    loss='sparse_categorical_crossentropy',  # Appropriate loss function for classification
    metrics=['accuracy']  # Monitor accuracy during training
)

# Set up early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Watch validation loss
    patience=3,  # Stop if no improvement for 3 epochs
    restore_best_weights=True  # Keep the best weights
)

# Train the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),  # Generate augmented training data
    validation_data=(x_val, y_val),  # Validation data to monitor performance
    epochs=10,  # Number of times to train on entire dataset
    callbacks=[early_stopping],
    verbose=1  # Show progress bar
)

# Function to process uploaded images to match MNIST format
def process_image(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to 28x28 to match MNIST format
    img = cv2.resize(img, (28, 28))

    # Handle inverted images (white on black)
    mean_value = np.mean(img)
    if mean_value < 127:
        img = cv2.bitwise_not(img)

    # Apply various image processing techniques to clean up the image
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.bitwise_not(img)

    # Normalize and reshape for model input
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Function to make predictions on uploaded images
def predict_digit(image_path):
    processed_img = process_image(image_path)
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)  # Get the digit with highest probability

    # Display results
    print(f"Predicted digit: {predicted_digit}")
    display(Image(filename=image_path))
    
    # Show probability for each digit
    print("\nProbabilities for each digit:")
    for i, prob in enumerate(prediction[0]):
        print(f"Digit {i}: {prob*100:.2f}%")

# Handle file upload and make prediction
print("\nPlease upload an image file:")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\nPredicting digit in {filename}:")
    predict_digit(filename)

# Plot training history to visualize model performance
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('mnist_model.h5')

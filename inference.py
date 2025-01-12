# Import required libraries
import tensorflow as tf  # Main deep learning library
from tensorflow import keras  # High-level neural network API
import cv2  # OpenCV library for image processing
import numpy as np  # For numerical operations
from IPython.display import Image, display  # For displaying images in notebook

# Load a previously trained model from a file named 'mnist_model.h5'
# This model was trained on the MNIST dataset and saved earlier
model = keras.models.load_model('mnist_model.h5')

# Function to process and prepare images for prediction
def process_image(image_path):
    # Read the image in grayscale mode (single channel)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to 28x28 pixels to match MNIST format
    img = cv2.resize(img, (28, 28))

    # Calculate average pixel value to determine if image is inverted
    mean_value = np.mean(img)
    # If mean value is less than 127 (darker), assume it's inverted
    if mean_value < 127:
        # Invert the colors (white becomes black and vice versa)
        img = cv2.bitwise_not(img)

    # Apply adaptive thresholding to improve image quality
    # This helps handle different lighting conditions and contrast levels
    # Parameters: max value=255, adaptive method=Gaussian, binary threshold, block size=11, constant=2
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Gaussian blur to reduce noise
    # Uses a 3x3 kernel
    img = cv2.GaussianBlur(img, (3,3), 0)

    # Normalize the image contrast to use full range of values (0-255)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Invert the image again to match MNIST format (black digits on white background)
    img = cv2.bitwise_not(img)

    # Convert to float32 and normalize pixel values to range 0-1
    img = img.astype('float32') / 255.0
    # Reshape image to match model's expected input shape: (1, 28, 28, 1)
    # 1: batch size, 28x28: image dimensions, 1: number of channels
    img = img.reshape(1, 28, 28, 1)
    return img

# Function to make predictions on processed images
def predict_digit(image_path):
    # Process the image using the function defined above
    processed_img = process_image(image_path)

    # Use the model to make a prediction
    # Returns array of probabilities for each digit (0-9)
    prediction = model.predict(processed_img)
    # Get the digit with highest probability
    predicted_digit = np.argmax(prediction)

    # Display the original image and the predicted digit
    print(f"Predicted digit: {predicted_digit}")
    display(Image(filename=image_path))

    # Show the probability for each possible digit (0-9)
    print("\nProbabilities for each digit:")
    for i, prob in enumerate(prediction[0]):
        # Convert probability to percentage and print
        print(f"Digit {i}: {prob*100:.2f}%")

# Set up file upload functionality (specific to Google Colab)
print("\nPlease upload an image file:")
from google.colab import files
uploaded = files.upload()

# Process each uploaded file
for filename in uploaded.keys():
    print(f"\nPredicting digit in {filename}:")
    predict_digit(filename)  # Make and display prediction for each uploaded image

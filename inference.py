import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from IPython.display import Image, display

# Load the saved model
model = keras.models.load_model('mnist_model.h5')

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))

    # Check if image is inverted (white on black)
    mean_value = np.mean(img)
    if mean_value < 127:
        img = cv2.bitwise_not(img)

    # Add adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Add noise reduction
    img = cv2.GaussianBlur(img, (3,3), 0)

    # Ensure proper contrast
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Invert again to match MNIST format (black digits on white background)
    img = cv2.bitwise_not(img)

    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_digit(image_path):
    # Process image
    processed_img = process_image(image_path)

    # Make prediction
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)

    # Display image and prediction
    print(f"Predicted digit: {predicted_digit}")
    display(Image(filename=image_path))

    # Show probabilities for all digits
    print("\nProbabilities for each digit:")
    for i, prob in enumerate(prediction[0]):
        print(f"Digit {i}: {prob*100:.2f}%")

# Test with uploaded image
print("\nPlease upload an image file:")
from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\nPredicting digit in {filename}:")
    predict_digit(filename)

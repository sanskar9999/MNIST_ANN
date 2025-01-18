https://medium.com/@mrearthwormwormling/machine-learning-basics-teaching-a-computer-to-read-handwritten-digits-da30d989aaa8

---

Machine Learning Basics: Teaching a Computer to Read Handwritten digits
Final resulting model's predictionsIn this post, we're going to dive into a simple and fun project, building a model that can recognize handwritten digits (0–9)
We'll use a famous dataset called MNIST (this is like the "Hello, World!" of image recognition in the ML world)

---

You might think, "Isn't this a solved problem?"

Well, Yes.
While digit recognition is a solved problem, it's a fantastic way to understand the basics of how machines learning works. 
Computers don't "see" like we do, They see images as grids of numbers
Each number represents the brightness of a pixel. For a black and white image 0 = black, 255 = white and everything in between is a shade of gray.
Our goal is to train a neural network, (fancy term for a computer program which mimics how neurons in the human brain work) to recognize patterns in these numbers that correspond to different digits.

---

Let's Get Started (with Code!)

We'll be using Python (because its easy) and a few powerful libraries:
TensorFlow/Keras: For building and training neural networks much easier by a providing a simplified API
Matplotlib: This will helps us visualize what's going on, like plotting graphs of our model's performance
OpenCV: Helps us clean up / prepare images for our model
NumPy: This is just a library for doing numerical operations in Python

1. Loading the MNIST Dataset

digit samples from MNIST datasetFirst, we get the MNIST dataset, it has over 70,000 images of handwritten digits. We split it into training data (used to teach our model) and testing data (to see how well it learned).
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
2. Normalizing the Data

We then "normalize" the pixel values. This just means converting those grey pixel values from a range of 0–255 to 0–1
for this, we just divide them by 255
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
3. Data Augmentation 

Real-world is messy.
People write with different handwriting styles, angles, varying pressure, and this makes everyone's digits look all different and unique. To prepare our model for this, we use a technique called data augmentation. To make our model more general.
So we create modified versions of our training images - rotate them, shift them, zoom them, and even inverted their colors. This brings all kinds of variety to our dataset, helping the model learn that a "2" is still a "2" even if it's a bit tilted or squished.
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    preprocessing_function=lambda x: 1 - x if np.random.random() > 0.5 else x
)
4. Building the Neural Network

Now the heart of our project: the neural network. 
We're using a simple feedforward network, which just means information flows in one direction, from input to output.
Our network has these layers:
Flatten: This is the first layer, It takes our 28x28 pixel image and turns it into a long list of 784 numbers. One neuron for each pixel.
Dense: These are the "thinking" layers. Each "neuron" in a dense layer is connected to all the neurons in the previous layer. We use the ReLU a popular activation function.
Batch Normalization: This layer helps stabilize and speed up training by normalizing the outputs of the previous layer.
Output (Dense): This layer has 10 neurons. One for each digit. We use the Softmax activation function, which turns the neuron outputs into probabilities (0 to 1). The neuron with the highest probability will be the model's guess for the digit.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])
5. Training the Model

We "train" the model by feeding it our training data and telling it the correct answer for each image after its guess. The model adjusts its weights to minimize the difference between its predictions and the correct answers.
We use the Adam optimizer (a popular algorithm for adjusting the learning rate for gradient descent) and the sparse categorical cross-entropy loss function (this just measures how wrong the model's predictions are to correct them)
We also use early stopping, which means we stop training if the model's performance stops improving. This prevents overfitting, where the model memorizes the training data because of too much training, so it doesn't generalize well to new images.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[early_stopping],
    verbose=1
)
6. Testing it 
Testing the Model on my own handwritten Image examplesAfter training, we can upload our own handwritten digit image, and the saved model will try to guess what it is. We use OpenCV to process the image, making it similar to the MNIST images the model was trained on.
def predict_digit(image_path):
    processed_img = process_image(image_path)
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    # ... (Display results) ...
uploaded = files.upload()
for filename in uploaded.keys():
    predict_digit(filename)
7. Visualizing the Results

Finally, we plot the model's accuracy and loss during training. This helps us see how well it learned and we can spot if there were any issues like overfitting.
Our Model's Accuracy and Loss over the Epoches

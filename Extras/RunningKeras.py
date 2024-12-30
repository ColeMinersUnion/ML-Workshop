import tensorflow as tf
from PIL import Image
import numpy as np

fdata = "MyMNIST.png"
fmodel = "tf_mnist_cnn.keras"

model = tf.keras.models.load_model(fmodel)

# Function to preprocess the image
def image_to_tensor(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')
    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Normalize the image
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_tensor):
    # Make a prediction by passing the image tensor to the model
    predictions = model.predict(image_tensor)
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

if __name__ == '__main__':
    model = tf.keras.models.load_model(fmodel)
    img_tensor = image_to_tensor(fdata)
    predicted_class = predict_image_class(model, img_tensor)
    print(f"\nPredicted Class: {predicted_class}\n")
import torch
from torch import nn
from torch.nn import functional as F

#! Runs the PyTorch CNN model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#These filenames should really be cli args but alas
fmodel = "mnist_cnn.pt"
fdata = "MyMNIST.png"

# Function to load the model
def load_model(model_path):
    model = Net()  # Use the correct architecture
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

#converting the image to tensor
def image_to_tensor(image_path):
    from PIL import Image
    import torchvision.transforms as transforms

    # Load the image
    img = Image.open(image_path)
    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)), # Resize the image to 28x28 pixels
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) #MNIST mean and std
    ])
    # Apply the transformations
    img_tensor = preprocess(img)
    return img_tensor

# Function to predict the class of an image
def predict_image_class(model, image_tensor):
    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze(0)
    # Make a prediction by passing the image tensor to the model
    with torch.no_grad():
        output = model(image_tensor)
        # Get the predicted class
        pred = output.argmax(dim=1, keepdim=True)
        return pred.item()
    

if __name__ == '__main__':
    model = load_model(fmodel)
    img_tensor = image_to_tensor(fdata)
    predicted_class = predict_image_class(model, img_tensor)
    print(f"\nPredicted Class: {predicted_class}\n")



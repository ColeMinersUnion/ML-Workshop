import torch
import torch.nn as nn #pytorch's neural network
import torch.nn.functional as F
import torch.optim as optim #optimization?
from torchvision import datasets, transforms #datasets is were MNIST comes from. 
from torch.optim.lr_scheduler import StepLR

class Network(nn.Module): 
    # Input size is the size of the input image
    # hidden size is the size of the hidden layer
    # num_classes is the number of output classes
    def __init__(self, input_size, hidden_size, num_classes):
        #Inherit nn.Module
        super(Network, self).__init__()
        
    #dessigning the neural network
    def forward(self, x):
        pass


def train(model, device, train_loader, optimizer, epoch):
    model.train() 
    return

def test(model, device, test_loader):
    model.eval()
    return
#Everything else

def main():
    # Load & Download the dataset
    dataset = datasets.MNIST(root='../data', train=True, download=True)

    # Compute the mean and standard deviation. 
    # The greyscale pixel values range from 0-255. 
    mean = dataset.data.float().mean() / 255
    std = dataset.data.float().std() / 255

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    #split the dataset into a training set and a testing set
    training_data = datasets.MNIST('../data', train=True, transform=transform)
    testing_data = datasets.MNIST('../data', train=False, transform=transform)

    #Next we create loaders. These are used to load the data in batches.
    train_batch_size = 64
    test_batch_size = 1000 #Mostly arbitrary

    #Arguments used to configure the loaders
    training_kwargs = {'batch_size': train_batch_size}
    testing_kwargs = {'batch_size': test_batch_size}

    #Possible Hardware Acceleration
    if torch.cuda.is_available():
        #! Note: I don't have CUDA, this code is untested. Was pulled from PyTorch Example.
        device = torch.device("cuda")
        cuda_kwargs = {'num_workers': 1, 
                       'pin_memory': True,
                       'shuffle': True}
        training_kwargs.update(cuda_kwargs)
        testing_kwargs.update(cuda_kwargs)
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")


    #Initialize the loaders
    # ** is an unpacking operator. Unpacks each item from the directory into the function as arguments. 
    train_loader = torch.utils.data.DataLoader(training_data,**training_kwargs)
    test_loader = torch.utils.data.DataLoader(testing_data, **testing_kwargs)
    




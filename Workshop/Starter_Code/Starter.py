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
    def __init__(self, input_size, hidden_layers, hidden_units, num_classes):
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
if __name__ == '__main__':
    pass

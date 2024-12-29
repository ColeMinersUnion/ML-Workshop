import torch
import torch.nn as nn #pytorch's neural network
import torch.nn.functional as F
import torch.optim as optim #optimization?
from torchvision import datasets, transforms #datasets is were MNIST comes from. 
from torch.optim.lr_scheduler import StepLR

class Network(nn.Module): 
    def __init__(self):
        #Inherit nn.Module
        super(Network, self).__init__()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train() 
    pass

def test(model, device, test_loader):
    model.eval()


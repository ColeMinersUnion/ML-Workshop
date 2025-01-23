import torch
import torch.nn as nn #pytorch's neural network
import torch.nn.functional as F
import torch.optim as optim #optimization?
from torchvision import datasets, transforms #datasets is were MNIST comes from. 
from torch.optim.lr_scheduler import StepLR #scheduling algorithm


from Network import Network
from Test import test
from Train import train
    


def main():
    #Parameters
    lr = 1.0
    gamma = 0.7 
    epochs = 15 #Mostly arbitrary
    save = False 
    hidden_layers = 3 #Also arbitrary

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

    input_size=dataset.data.shape[1] * dataset.data.shape[2] #784 = 28*28
    model = Network(input_size=input_size, hidden_layers=hidden_layers, hidden_units=[256]).to(device)

    #Create an optimizer
    #Certainly one of the optimizers
    optimizer = optim.Adadelta(model.parameters(), lr=lr) 

    #Scheduling
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if save:
        torch.save(model.state_dict(), "../Extras/mnist_dnn.pt")

if __name__ == '__main__':
    main()
    #please don't break
    #update it didn't break






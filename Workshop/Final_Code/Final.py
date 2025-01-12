import torch
import torch.nn as nn #pytorch's neural network
import torch.nn.functional as F
import torch.optim as optim #optimization?
from torchvision import datasets, transforms #datasets is were MNIST comes from. 
from torch.optim.lr_scheduler import StepLR #scheduling algorithm

#Implementing a dense neural network
class Network(nn.Module): 
    
    #TODO: Unpack hidden_units to hidden_layers
    def __init__(self, input_size, hidden_layers=1, hidden_units = [128], output_size=10):
        #Inherit nn.Module
        super(Network, self).__init__()
        
        #Define the layers
        layers = []

        # There must be at least one hidden layer
        if (hidden_layers < 1):
            from warnings import warn
            warn("There must be at least one hidden layer. Setting hidden_layers to 1.")
            hidden_layers = 1
        elif (hidden_layers != len(hidden_units)): 
            from warnings import warn
            warn("The number of hidden units does not match the number of hidden layers. Using the first value in hidden_units for all layers.")
        
        if (hidden_layers != 1 and len(hidden_units) == 1):
            hidden_units = tuple([hidden_units[0]] * (hidden_layers + 1))
        
        layers.append(nn.Linear(input_size, hidden_units[0]))
        layers.append(nn.ReLU())
        
        # Add additional hidden layers if specified
        for i in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units[i+1], hidden_units[i+2]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_units[-1], output_size))
        
        # Use nn.Sequential to stack the layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input
        x = self.model(x)
        output = F.log_softmax(x, dim=1) #! Use log_softmax
        return output



def train(model, device, train_loader, optimizer, epoch):
    log_interval = 10
    model.train() 
    criterion = nn.CrossEntropyLoss()  # Set criterion
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # Use CrossEntropyLoss
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Use CrossEntropyLoss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    


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






import torch
import torch.nn as nn #pytorch's neural network
import torch.nn.functional as F


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
            hidden_layers = 1
        if (hidden_layers != 1 and len(hidden_units) == 1):
            hidden_units = tuple([hidden_units[0]] * (hidden_layers + 1))
        
                            #Current layer size, Next layer size
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
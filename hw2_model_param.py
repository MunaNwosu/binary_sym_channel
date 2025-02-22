import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)  # MNIST: Input size is 28*28=784
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)  # Dropout with 20% probability
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.softmax(self.fc3(x), dim=1)  # Softmax for 10-way classification
        return x

# Initialize the neural network with He initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

# Transmiting bits over Binary symetrical channel(BMC)  
def transmit_parameters(parameters, error_probability):
    transmitted_parameters = [p + torch.randn_like(p) * error_probability for p in parameters]
    return transmitted_parameters

# Create data loaders for MNIST
batch_size = 32
mnist_loader = DataLoader(datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True),
    batch_size=batch_size, shuffle=True)

# Create and initialize the model
model = Net()
model.apply(initialize_weights)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(mnist_loader): 
        optimizer.zero_grad()  # Zero the gradients
        data = data.view(data.size(0), -1)  # Flatten the input data
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Calculate the loss
        loss.backward()  # Back propagation
        optimizer.step()  # Update the weights

    print('Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))


# Evaluation on the test set
model.eval() 

correct = 0
total = 0

with torch.no_grad():
    for data, target in mnist_loader:
        data = data.view(data.size(0), -1)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

"""
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
"""

total_bits  = sum(p.numel() * p.element_size() * 8 for p in model.parameters())
print(f'Total number of bits to be transmitted: {total_bits}\n')
print('Accuracy on the test set before BSC : {:.2f}%\n'.format(100 * correct / total))


error_probability  = [10**-6, 10**-4, 10**-2, 10**-1]

#Binary Symmetry Channel with error probabilities
for error in error_probability:
    print("Error probability:  %f" % error)

    transmitted_parameters  = transmit_parameters(model.parameters(), error)

    # Update the model with the received parameters
    with torch.no_grad():
        for param, received_param in zip(model.parameters(), transmitted_parameters):
            param.copy_(received_param)
    
    model.eval()
    correct = 0
    total = 0  
    
    with torch.no_grad():
        for data, target in mnist_loader:  #error free test data
            data = data.view(data.size(0), -1)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on the test set after BSC : {:.4f}%\n'.format(100 * correct / total))
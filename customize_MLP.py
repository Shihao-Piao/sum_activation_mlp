import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 64
learning_rate = 0.01 #0.01
epochs = 20

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))  # flatten from (1, 28, 28) -> (784,)
])

train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size)


class ActivatedWeightedSum(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU()):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = activation

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = x.unsqueeze(1)                      # -> (batch_size, 1, input_dim)
        w = self.weight.unsqueeze(0)            # -> (1, output_dim, input_dim)
        out = self.activation(x * w)            # -> (batch_size, output_dim, input_dim)
        out = out.sum(dim=2) + self.bias
        return out                              # -> (batch_size, output_dim)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 10)  # output layer
        )

    def forward(self, x):
        return self.layers(x)
class activated_sum_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ActivatedWeightedSum(784,64),
            ActivatedWeightedSum(64, 10),
            #nn.Linear(64, 10)  # output layer
        )
        self.out_layer =nn.Linear(64, 10)

    def forward(self, x):
        x = self.layers(x)
        #print(x)
        #x = self.out_layer(x)
        return x


#model = MLP()
model = activated_sum_MLP()
#test_model = ActivatedWeightedSum(784,128)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

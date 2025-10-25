import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sklearn

import numpy as np

BATCH_SIZE = 32
ALPHA  = 0.2
EPOCHS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = torchvision.datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),

        # Resize from 28x28 to 32x32 images 
        v2.Resize((32, 32)),

        v2.ToDtype(torch.float32, scale=True),

        # Achieve a value of -0.1 for white pixels and 1.175 for black pixels as described in the paper
        v2.Lambda(lambda x: 1.275*x - 0.1)
    ]),
    # One-hot encode labels
    target_transform=lambda y: torch.nn.functional.one_hot(torch.tensor(y), num_classes=10).float()
)

test_data = torchvision.datasets.MNIST(
    root='data',
    train=False,
    download = True,
    transform=v2.Compose([
        v2.ToImage(),

        # Resize from 28x28 to 32x32 images 
        v2.Resize((32, 32)),

        v2.ToDtype(torch.float32, scale=True),

        # Achieve a value of -0.1 for white pixels and 1.175 for black pixels as described in the paper
        v2.Lambda(lambda x: 1.275*x - 0.1)
    ]),
    # One-hot encode labels
    target_transform=lambda y: torch.nn.functional.one_hot(torch.tensor(y), num_classes=10).float()
)

train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.s2 = SubSampling(in_channels=6)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = SubSampling(in_channels=16)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)
        self.tanh = nn.Tanh()
    
    def forward(self, x):        
        return self.tanh(self.out(self.tanh(self.f6(self.flatten(self.tanh(self.c5(self.s4(self.tanh(self.c3(self.s2(self.tanh(self.c1(x)))))))))))))

# Define the SubSampling layer as per LeNet-5 architecture
class SubSampling(nn.Module):
    def __init__(self, in_channels, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):
        # Apply average pooling
        x = self.pool(x)
        # Apply learnable weight and bias
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        # Apply tanh activation
        return torch.tanh(x)
    
# Initialize weights using LeCun normal initialization
def lecun_normal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # Linear layers
        fan_in = m.weight.data.size(1) 
        # Convolutional layers
        if isinstance(m, nn.Conv2d):
            fan_in *= m.weight.data[0][0].numel() 

        std = 1. / np.sqrt(fan_in)
        nn.init.normal_(m.weight, mean=0, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

m = LeNet5().to(device)
m.apply(lecun_normal_init)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(m.parameters(), lr=ALPHA)

losses = []
accuracies = []

for epoch in tqdm(range(EPOCHS)):
    print(f'Epoch {epoch+1}/{EPOCHS}')

    m.train()
    train_loss = 0
    train_acc = 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = m(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_dataloader)
    train_acc /= len(train_data)
    losses.append(train_loss)
    accuracies.append(train_acc)

    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}')

m.eval()
y_preds = []
y_true = []
with torch.no_grad():
    test_loss = 0
    test_acc = 0
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = m(X)

        y_true.extend(y.argmax(dim=1))
        y_preds.extend(y_pred.argmax(dim=1))

        test_loss += loss_fn(y_pred, y)
        test_acc += ((y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item())

    test_loss /= len(test_dataloader)
    test_acc /= len(test_data)

    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')


# Plot training loss and accuracy
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].plot(range(1, EPOCHS+1), losses, label='MSE Loss', color='blue')
ax[0].set_title('Training Loss over Epochs')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[1].plot(range(1, EPOCHS+1), accuracies, label='Accuracy', color='orange')
ax[1].set_title('Training Accuracy over Epochs')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
plt.savefig('results/metrics.png')

# Visualize predictions
indices = torch.randint(0, len(test_data), (16,))
fig, ax = plt.subplots(4, 4, figsize=(8, 8))

for i, idx in enumerate(indices):
    img, label = test_data[idx]
    with torch.no_grad():
        output = m(img.unsqueeze(0).to(device))
        predicted_label = output.argmax(dim=1).item()
    
    if predicted_label == label.argmax().item():
        color = 'green'
    else:
        color = 'red'
    
    ax[i//4, i%4].imshow(img.squeeze(), cmap='gray')
    ax[i//4, i%4].set_title(f'True: {label.argmax().item()} | Predicted: {predicted_label}', c=color)
    ax[i//4, i%4].axis('off')

plt.tight_layout()
plt.savefig('results/predictions.png')

# Plot confusion matrix
conf_mat = sklearn.metrics.confusion_matrix(y_true=[y.cpu() for y in y_true],
                                            y_pred=[y.cpu() for y in y_preds])
plt.figure(figsize=(10, 8))
sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png') 

# Print classification report
print(sklearn.metrics.classification_report(
    [y.cpu() for y in y_true],
    [y.cpu() for y in y_preds],
    digits=4
))

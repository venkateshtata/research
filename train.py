import torch
import torch.nn as nn
import torch.optim as optim
from data_gen import drivingDataset
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import tqdm

device = 'cpu'

lr = 1e-4
weight_decay = 1e-5
batch_size = 32
num_workers = 8
test_size = 0.8
shuffle = True

epochs = 80

torch.set_default_dtype(torch.float64)

class DriverNet(nn.Module):

  def __init__(self):
        super(DriverNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*13*33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

  def forward(self, x):
      x = x.view(x.size(0), 3, 160, 320)
      #print("x.size(0) : ", x.size(0))
      output = self.conv_layers(x)
      #print("output from conv layers : ", output.shape)
      output = output.view(output.size(0), -1)
      #print("output dims : ", output.shape)
      output = self.linear_layers(output)
      return output

# Define model
print("==> Initialize model ...")
model = DriverNet()
print("==> Initialize model done ...")




transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

dataset = drivingDataset(transform = transformations)

train_set, test_set = torch.utils.data.random_split(dataset, [14526, 3651])

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=shuffle)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=shuffle)

# model = torchvision.models.googlenet(pretrained=True)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

loss = 0
for i in range(epochs):
    
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):

        images = data.to(device = device)
        angles = targets.to(device = device)

        angles = angles.view(-1,1)
        
        optimizer.zero_grad()

        predicted_angles = model(images)

        #print("angles shape : ", angles.shape)
        #print("predicted angles shape : ", predicted_angles.shape)

        #print("angle : ", angles[10])
        #print("predicted angle : ", predicted_angles[10])

        loss = criterion(predicted_angles, angles)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if(i%10==9):
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/10))
            running_loss = 0.0

    if(epoch % 5 == 0):
        torch.save(model.state_dict(), "trained_weights.pt")




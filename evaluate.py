import torch
import torch.nn as nn
import torch.optim as optim
from data_gen import drivingDataset
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('tests/driving')

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
model.load_state_dict(torch.load("trained_weights.pt"))
model.eval()
print("==> Initialize model done ...")


transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

dataset = drivingDataset(transform = transformations)

train_set, test_set = torch.utils.data.random_split(dataset, [14526, 3651])


train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=shuffle)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=shuffle)

dataiter = iter(train_loader)
images, lab = dataiter.next()

writer.add_graph(model, images)

model.to(device)

count = 0
for batch_idx, (data, targets) in enumerate(train_loader):
    images = data.to(device = device)
    angles = targets.to(device = device)
    
    if(count>=1000):
        break
    for i in range(len(images)):
        angle = angles[i]
        image = images[i]
        image = torch.unsqueeze(image, 0)
        #print("angle shape : ", angle.shape)
        #print("angle : ", angle)
        #print("image shape : ", images[i][0].shape)
        #print("image : ", images[i])
        predicted_angle = model(image)
        writer.add_scalars('angles vs predicted_angles', {'angle':angle, 'predicted_angle':predicted_angle}, count)
        print(count)
        count+=1

writer.close()



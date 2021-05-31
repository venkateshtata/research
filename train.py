import torch
import torch.nn as nn
import torch.optim as optim
from data_gen import drivingDataset
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/driving')

device = 'cpu'

lr = 1e-5
weight_decay = 1e-5
batch_size = 32
num_workers = 8
test_size = 0.8
shuffle = True

epochs = 100

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
            nn.Linear(in_features=64*13*33, out_features=1164),
            nn.ELU(),
            nn.Linear(in_features=1164, out_features=100),
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

dataiter = iter(train_loader)
images, lab = dataiter.next()

writer.add_graph(model, images)
writer.close()

# model = torchvision.models.googlenet(pretrained=True)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(epochs):
    
    train_loss = 0.0
    valid_loss = 0.0
    

    # Model Training
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

        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        #writer.add_scalar('training loss', train_loss, epoch * len(train_loader) + batch_idx)
        
        '''
        train_loss += loss.item()
        

        if(batch_idx%30==29):

            writer.add_scalar('training loss', running_loss / 30, epoch * len(train_loader) + batch_idx)

            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/30))
            running_loss = 0.0
        '''
        if(batch_idx%30==29):
            writer.add_scalar('training loss', train_loss/30, epoch * len(train_loader) + batch_idx)

    # Model Validation
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device = device)
        target = target.to(device = device)
            
        target = target.view(-1,1)
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        # calculate the batch loss
        loss = criterion(output, target)
        
        # update average validation loss
        valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        #writer.add_scalar('validation loss', valid_loss, epoch * len(test_loader) + batch_idx)

        if(batch_idx%30==29):
            writer.add_scalar('validation loss', valid_loss/30, epoch * len(test_loader) + batch_idx)

    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(test_loader.dataset)

    writer.add_scalar('Epoch Train Loss', train_loss/30, epoch)
    writer.add_scalar('Epoch Valid Loss', valid_loss/30, epoch)
    
    '''
    writer.add_scalar('training loss', train_loss)
    writer.add_scalar('validation loss', valid_loss)
    '''

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

    #torch.save(model.state_dict(), "trained_weights.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, "trained_weights2.pth")




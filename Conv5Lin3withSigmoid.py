import torch.nn as nn
import torch.nn.functional as F
import torch
class Conv5Lin3withSigmoid(nn.Module):
    def __init__(self):
        super(Conv5Lin3withSigmoid, self).__init__()
    #IN: 120x120, OUT: 60x60
        self.layer1 = nn.Sequential(
            nn.Conv2d(14, 28, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        #IN: 60x60, OUT: 30x30
        self.layer2 = nn.Sequential(
            nn.Conv2d(28, 56, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        #IN: 30x30, OUt: 15x15
        self.layer3 = nn.Sequential(
            nn.Conv2d(56, 112, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        #IN: 15x15, OUT: 7x7
        self.layer4 = nn.Sequential(
            nn.Conv2d(112, 224, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        #IN: 7x7, OUT: 3x3
        self.layer5 = nn.Sequential(
            nn.Conv2d(224, 224, kernel_size=5),
            nn.ReLU()
        )
        
        
        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(2016,1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        #x = x.view(-1,10976)
        x = self.drop_out(x)
        x = F.relu(self.fc1(x))
        x = self.drop_out(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        x = self.fc3(x)

        return torch.sigmoid(x)
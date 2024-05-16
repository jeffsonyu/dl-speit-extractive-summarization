import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassfier(nn.Module):
    def __init__(self, nn_channels):
        super(MLPClassfier, self).__init__()
        
        self.nn_channels = nn_channels
        self.fc_list = nn.ModuleList([
            nn.Linear(self.nn_channels[i], self.nn_channels[i+1]) for i in range(len(self.nn_channels)-1)
        ])
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        for i in range(len(self.nn_channels)-2):
            x = F.relu(self.fc_list[i](x))
            x = self.dropout(x)
        
        x = self.fc_list[-1](x)
        
        return x

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(768*32//8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        return x
    
if __name__ == "__main__":
    model = MLPClassfier([768, 128, 32, 16, 2]).to("cuda")
    x = torch.randn(32, 768).to("cuda")
    print(model(x).size())
    
    cnn = CNNClassifier().to("cuda")
    x = torch.randn(32, 768).to("cuda")
    print(cnn(x).size())
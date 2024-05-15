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
    
if __name__ == "__main__":
    model = MLPClassfier([768, 128, 32, 16, 2]).to("cuda")
    x = torch.randn(32, 768).to("cuda")
    print(model(x).size())
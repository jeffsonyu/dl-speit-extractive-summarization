import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassfier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassfier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    model = MLPClassfier(768, 128, 2).to("cuda")
    x = torch.randn(32, 768).to("cuda")
    print(model(x).size())
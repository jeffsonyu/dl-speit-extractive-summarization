import torch
from torch import nn

criterion = nn.CrossEntropyLoss()
y_pred = torch.tensor([0, 1], dtype=torch.float32)
y = torch.tensor(0, dtype=torch.long)
print(criterion(y_pred, y).item())
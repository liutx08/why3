import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # 这里是个简单的线性层示例

    def forward(self, x):
        return self.fc(x)
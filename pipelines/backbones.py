import torch
import torch.nn as nn
class MLPBackbone(nn.Module):
    def __init__(self, num_layers, input_size, output_size, hidden_size):
        super(MLPBackbone, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class CNNBackbone(nn.Module):
    def __init__(self, num_layers, input_size, input_channels, output_size, num_filters, kernel_size):
        super(CNNBackbone, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_channels, num_filters, kernel_size, padding = 'same'))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.MaxPool1d(2))
            input_channels = num_filters
            num_filters *= 2
        
        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(int(num_filters/2) * (input_size // (2 ** num_layers)), output_size)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN2DBackbone(nn.Module):
    def __init__(self, num_layers, input_size, input_channels, output_size, num_filters, kernel_size):
        super(CNN2DBackbone, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, num_filters, kernel_size, padding = 'same'))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.MaxPool2d(2))
            input_channels = num_filters
            num_filters *= 2
        
        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(int(num_filters/2), output_size)
    
    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.model(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x
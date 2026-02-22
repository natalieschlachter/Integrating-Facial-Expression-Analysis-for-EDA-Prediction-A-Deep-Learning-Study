import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Openface(nn.Module):
    def __init__(self):
        super(CNN_Openface, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
        
        self.landmark_mlp = nn.Sequential(
            nn.Linear(136, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size=512 + 64, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(256, 128)  
        self.fc2 = nn.Linear(128, 1)    
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, images, landmarks):
   
        B, T, C, H, W = images.size()
        
        images = images.view(B * T, C, H, W)
        landmarks = landmarks.view(B * T, 136)
        
        x = F.leaky_relu(self.bn1(self.conv1(images)))  
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))  
        
        x = F.leaky_relu(self.bn5(self.conv5(x)))
    
        x = self.global_avg_pool(x)
        x = x.view(B * T, -1)      
        
        l = self.landmark_mlp(landmarks) 
        
        combined = torch.cat((x, l), dim=1)  
        
        combined = combined.view(B, T, -1)  
        
        lstm_out, (h_n, c_n) = self.lstm(combined)
        
        lstm_last = lstm_out[:, -1, :]  
    
        x = F.leaky_relu(self.fc1(lstm_last))  
        x = self.dropout(x)
        output = self.fc2(x)                  
        
        return output

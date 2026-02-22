import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, cnn_feature_extractor, hidden_size=128):
        super(CNNLSTM, self).__init__()
        self.cnn = cnn_feature_extractor 

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                            batch_first=True)  
    
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
       
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)

        features = self.cnn(x)  # (B*T, 512)

        features = features.view(B, T, -1)  # (B, T, 512)

        lstm_out, (h_n, c_n) = self.lstm(features)

                                           
        predictions = self.fc(lstm_out)  #    final shape: (B, T, 1)

        return predictions  # (B, T, 1)


class ComplexCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(ComplexCNNFeatureExtractor, self).__init__()
    
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

    def forward(self, x):
        
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        
        x = self.global_avg_pool(x)    # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)      # (batch, 512)
        return x


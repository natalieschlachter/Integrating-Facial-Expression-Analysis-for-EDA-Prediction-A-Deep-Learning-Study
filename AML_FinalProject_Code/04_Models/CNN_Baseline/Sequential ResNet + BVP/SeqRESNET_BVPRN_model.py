import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet34", pretrained=True, freeze=True):
        """
        model_name: can be 'resnet18', 'resnet34', 'resnet50', etc.
        pretrained: if True, load ImageNet weights
        freeze:     if True, freeze backbone weights
        """
        super(ResNetFeatureExtractor, self).__init__()

        # 1) Load a pre-trained ResNet
        if model_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            out_dim = 512
        elif model_name == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            out_dim = 512
        elif model_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            out_dim = 2048
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # 2) Remove the final fully-connected classification layer
        #    We only want to keep the layers up to (and excluding) backbone.fc
        #    backbone.children() => list of modules in the order:
        #    [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = out_dim

        # 3) Optionally freeze backbone weights
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: (B*T, 3, H, W)
        returns: (B*T, out_dim)
        """
        # Pass images through the ResNet backbone
        x = self.backbone(x)  # => (B*T, out_dim, 1, 1)
        x = x.view(x.size(0), -1)  # => (B*T, out_dim)
        return x


class CNNLSTMWithBVP(nn.Module):
    def __init__(self, feature_extractor, hidden_size, bvp_embed_size=32):
        """
        feature_extractor: an instance of ResNetFeatureExtractor (or any CNN).
        hidden_size:       LSTM hidden dimension
        bvp_embed_size:    dimension to embed BVP values
        """
        super(CNNLSTMWithBVP, self).__init__()
        self.cnn = feature_extractor  # e.g. ResNetFeatureExtractor

        # Small embedding layer for per-frame BVP
        self.bvp_embedding = nn.Linear(1, bvp_embed_size)

        # LSTM input size = CNN out_dim + bvp_embed_size
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_dim + bvp_embed_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=0.2)

        
        # Final layer for EDA prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, bvp):
        """
        x shape:   (B, T, 3, H, W)
        bvp shape: (B, T)
        returns:   (B, T, 1) => EDA predictions
        """
        B, T, C, H, W = x.shape

        # 1) Flatten (B, T) => (B*T) so we can feed frames into ResNet
        x = x.view(B*T, C, H, W)                # => (B*T, 3, H, W)
        features = self.cnn(x)                  # => (B*T, out_dim)
        features = features.view(B, T, -1)      # => (B, T, out_dim)

        # 2) Embed BVP
        bvp = bvp.unsqueeze(-1).float()         # => (B, T, 1)
        bvp_embed = F.relu(self.bvp_embedding(bvp))  # => (B, T, bvp_embed_size)

        # 3) Concatenate CNN features and BVP embed => (B, T, out_dim + bvp_embed_size)
        combined_features = torch.cat([features, bvp_embed], dim=-1)

        # 4) LSTM
        outputs, _ = self.lstm(combined_features)  # => (B, T, hidden_size)

        outputs = self.dropout(outputs)        # => (B, T, hidden_size)

        # 5) EDA predictions => (B, T, 1)
        predictions = self.fc(outputs)
        return predictions

"""DeepPhys - 2D Convolutional Attention Network.
DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks
ECCV, 2018
Weixuan Chen, Daniel McDuff

Code from Github: https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/model/DeepPhys.py
"""

import torch
import torch.nn as nn

# Define a custom module for Attention Mask
class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        # The purpose of this is to normalize the input x by summing over specific dimensions
        # and scaling it appropriately for attention.
        # Sum across dimensions 2 and 3 (height and width of the input image)
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        # Return the normalized attention mask based on the shape of x
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        # Placeholder method to get configuration if needed
        config = super(Attention_mask, self).get_config()
        return config


class DeepPhys(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36):
        """Initialization of the DeepPhys model.

        Args:
          in_channels: the number of input channels. Default: 3 (RGB image)
          nb_filters1, nb_filters2: number of filters for the convolutional layers.
          kernel_size: size of the convolution kernel.
          dropout_rate1, dropout_rate2: dropout rates applied to different layers to reduce overfitting.
          pool_size: size of the pooling kernel (typically 2x2).
          nb_dense: the number of units in the fully connected layers.
          img_size: height/width of each frame, typically 36, 72, or 96 pixels.
        """
        super(DeepPhys, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense

        # Motion branch convolutions ->  motion input (differences in consecutive frames)
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Appearance branch convolutions -> raw input 
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Attention layers (these are used to apply attention masks to the features)
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        # Avg pooling layers to reduce the spatial dimensions of feature maps
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)

        # Dropout layers to prevent overfitting
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense layers (fully connected layers)
        # The number of units in the final dense layer depends on the image size (36, 72, or 96)
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')

        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        """
        Forward pass through the network.

        Args:
          inputs: A tensor of shape (batch_size, channels, height, width) for the input video frames.
          params: Any additional parameters for the forward pass (currently not used).
        
        Returns:
          out: The predicted output, typically the physiological measurement (e.g., heart rate).
        """

        # Split the input into two branches: diff_input (motion-related) and raw_input (appearance-related)
        diff_input = inputs[:, :3, :, :]  # Take first 3 channels - motion input (e.g., motion information)
        raw_input = inputs[:, 3:, :, :]   # Take next channels - raw appearance input (e.g., pixel values)

        # Process motion branch (diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))

        # Process appearance branch (raw_input)
        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        # Attention gating for the motion features based on appearance features
        g1 = torch.sigmoid(self.apperance_att_conv1(r2))  # Apply attention conv1
        g1 = self.attn_mask_1(g1)  # Apply attention mask to normalize attention
        gated1 = d2 * g1  # Element-wise multiplication with the motion features

        # Apply average pooling and dropout to reduce dimensions and prevent overfitting
        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        # Continue processing motion features (d4 -> d6)
        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        # Continue processing appearance features (r4 -> r6)
        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        # Attention gating for the appearance features based on motion features
        g2 = torch.sigmoid(self.apperance_att_conv2(r6))  # Apply attention conv2
        g2 = self.attn_mask_2(g2)  # Apply attention mask to normalize attention
        gated2 = d6 * g2  # Element-wise multiplication with the appearance features

        # Final pooling, dropout, and flattening
        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)

        # Flatten the output from 4D to 2D (batch_size, features)
        d9 = d8.view(d8.size(0), -1)

        # Fully connected layers for final output
        d10 = torch.relu(self.final_dense_1(d9))  # First dense layer # output to the range (−1, 1)
        d11 = self.dropout_4(d10)  # Dropout
        out = self.final_dense_2(d11)  # Final output (predicted physiological measurement)

        return out


    
# Create dummy input with 6 channels
#dummy_input = torch.randn(64, 6, 72, 72)  # Batch of 64, 6 channels, 72x72

# Initialize the model
#model = DeepPhys(in_channels=6, img_size=72)

# Forward pass
#output = model(dummy_input)
#print(f"Output shape: {output.shape}")  # Should print: Output shape: [64, 1]

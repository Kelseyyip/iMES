import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Module):        # For ICCN feature extraction
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
		    		# nn.Tanh()
                )
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Dropout(0.1),
                    nn.ReLU()
                    
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x_in = x
            x = layer(x)
            a = 1
        return x

class Decoder(nn.Module):       # Reconstruction of modality features. To improve ICCA
    def __init__(self, cca_size, hidden_size, input_size1):
        super(Decoder, self).__init__()
        self.layer_1_1 = nn.Linear(cca_size, 2 * hidden_size)
        self.layer_1_2 = nn.Linear(2 * hidden_size, input_size1)

    def forward(self, ccaoutput):
        rec = self.layer_1_2(self.layer_1_1(ccaoutput))
       
        return rec

### CNN model for
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim, input_height, input_width):
        super(CNN, self).__init__()
        input_channels = 1
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 尺寸缩小 1/2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)  # 尺寸再缩小 1/2
        )

        # 计算池化后的尺寸
        reduced_height = input_height // 4
        reduced_width = input_width // 4
        self.fc = nn.Linear(64 * reduced_height * reduced_width, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




class DeepCCA(nn.Module):       # Deep CCA model. MLP for feature extraction, then calculate CCA bettwen featuers
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss = cca_loss(outdim_size, device).loss
        self.mseloss = torch.nn.MSELoss()

    def forward(self, x1, x2):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2

class DeepCCA_updated(nn.Module):       # Deep CCA model. MLP for feature extraction, then calculate CCA bettwen featuers
    def __init__(self, input_channels, output_dim, input_height0, input_width0, input_height1, input_width1, device=torch.device('cpu')):
        super(DeepCCA_updated, self).__init__()

        self.model1 = CNN(input_channels, output_dim, input_height0, input_width0).double()
        self.model2 = CNN(input_channels, output_dim, input_height1, input_width1).double()

        self.loss = cca_loss(output_dim, device).loss
        self.mseloss = torch.nn.MSELoss()

    def forward(self, x1, x2):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2

import tensorly as tl
import torch
import torch.nn as nn
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from torch import diag

def mat_pow(mat, pow_, epsilon):
    # Computing matrix to the power of pow (pow can be negative as well)
    [D, V] = torch.linalg.eigh(mat)
    mat_pow = V @ diag((D + epsilon).pow(pow_)) @ V.T
    mat_pow[mat_pow != mat_pow] = epsilon  # For stability
    return mat_pow
def _demean(*views):
    return tuple([view - view.mean(dim=0) for view in views])

class cca_loss():
    def __init__(self, latent_dims: int, device, r: float = 0, eps: float = 1e-3):
        """
        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps
        self.device = device


    def loss(self, H1, H2):
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        n = H1.shape[0]

        H1bar, H2bar = _demean(H1, H2)

        SigmaHat12 = (1.0 / (n - 1)) * H1bar.T @ H2bar
        SigmaHat11 = (1 - self.r) * (
            1.0 / (n - 1)
        ) * H1bar.T @ H1bar + self.r * torch.eye(o1, device=H1.device)
        SigmaHat22 = (1 - self.r) * (
            1.0 / (n - 1)
        ) * H2bar.T @ H2bar + self.r * torch.eye(o2, device=H2.device)

        SigmaHat11RootInv = mat_pow(SigmaHat11, -0.5, self.eps)
        SigmaHat22RootInv = mat_pow(SigmaHat22, -0.5, self.eps)

        Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
        trace_TT = Tval.T @ Tval
        eigvals = torch.linalg.eigvalsh(trace_TT)

        # leaky relu encourages the gradient to be driven by positively correlated dimensions while also encouraging
        # dimensions associated with spurious negative correlations to become more positive
        eigvals = eigvals[torch.gt(eigvals, self.eps)]
        corr = torch.sum(torch.sqrt(eigvals))

        return -corr


class Decoder(nn.Module):  # Reconstruction of modality features. To improve ICCA
    def __init__(self, cca_size, hidden_size, input_size1):
        super(Decoder, self).__init__()
        self.layer_1_1 = nn.Linear(cca_size, 2 * hidden_size)
        self.layer_1_2 = nn.Linear(2 * hidden_size, input_size1)

    def forward(self, ccaoutput):
        rec = self.layer_1_2(self.layer_1_1(ccaoutput))

        return rec


### CNN model
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

class ICCN(nn.Module):       # Deep CCA model. MLP for feature extraction, then calculate CCA bettwen featuers
    def __init__(self, input_channels, output_dim, input_height0, input_width0, input_height1, input_width1, device=torch.device('cpu')):
        super(ICCN, self).__init__()

        self.model1 = CNN(input_channels, output_dim, input_height0, input_width0).double()
        self.model2 = CNN(input_channels, output_dim, input_height1, input_width1).double()

        self.loss = cca_loss(output_dim, device).loss

    def forward(self, x1, x2):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2





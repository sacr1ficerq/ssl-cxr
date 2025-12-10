import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_resnet18_cifar(projection_dim: int = 128) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class VICRegProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048, output_dim: int = 2048):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    def __init__(self, encoder_dim: int = 512, projection_dim: int = 128, temperature: float = 0.5):
        super().__init__()
        self.encoder = get_resnet18_cifar()
        self.encoder.fc = nn.Identity()
        self.projection_head = SimCLRProjectionHead(encoder_dim, 2048, projection_dim)
        self.temperature = temperature

    def forward(self, view1, view2):
        z1 = self.projection_head(self.encoder(view1))
        z2 = self.projection_head(self.encoder(view2))

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)

        sim_matrix = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        positive_mask = torch.zeros_like(mask)
        positive_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        positive_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=z.device)

        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), 
                           torch.arange(batch_size)], dim=0).to(z.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class VICReg(nn.Module):
    def __init__(self, encoder_dim: int = 512, projection_dim: int = 2048,
                 lambda_inv: float = 25.0, lambda_var: float = 25.0, lambda_cov: float = 1.0):
        super().__init__()
        self.encoder = get_resnet18_cifar()
        self.encoder.fc = nn.Identity()
        self.projection_head = VICRegProjectionHead(encoder_dim, 2048, projection_dim)
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(self, view1, view2):
        z1 = self.projection_head(self.encoder(view1))
        z2 = self.projection_head(self.encoder(view2))

        inv_loss = F.mse_loss(z1, z2)
        var_loss = self.variance_loss(z1) + self.variance_loss(z2)
        cov_loss = self.covariance_loss(z1) + self.covariance_loss(z2)

        loss = self.lambda_inv * inv_loss + self.lambda_var * var_loss + self.lambda_cov * cov_loss
        return loss

    def variance_loss(self, z):
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        loss = torch.mean(F.relu(1 - std))
        return loss

    def covariance_loss(self, z):
        batch_size, dim = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (batch_size - 1)
        off_diagonal = cov.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
        loss = off_diagonal.pow(2).sum() / dim
        return loss


class SSLModelWithProbe(nn.Module):
    def __init__(self, ssl_model, num_classes, encoder_dim=512):
        super().__init__()
        self.ssl_model = ssl_model
        self.linear_probe = nn.Linear(encoder_dim, num_classes)

    def forward(self, view1, view2, compute_ssl=True):
        h1 = self.ssl_model.encoder(view1)
        h2 = self.ssl_model.encoder(view2)

        if compute_ssl:
            z1 = self.ssl_model.projection_head(h1)
            z2 = self.ssl_model.projection_head(h2)
            return h1, h2, z1, z2
        return h1, h2


def get_resnet18_medmnist(pretrained=None, encoder_dim=512):
    if pretrained == 'imagenet':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif pretrained == 'imagenet_ssl':
        raise NotImplementedError("Load solo-learn checkpoint manually")
    else:
        model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Identity()
    return model


class SimCLRMedMNIST(nn.Module):
    def __init__(self, encoder_dim=512, projection_dim=128, temperature=0.5, pretrained=None):
        super().__init__()
        self.encoder = get_resnet18_medmnist(pretrained=pretrained, encoder_dim=encoder_dim)
        self.projection_head = SimCLRProjectionHead(encoder_dim, 2048, projection_dim)
        self.temperature = temperature

    def forward(self, view1, view2):
        z1 = self.projection_head(self.encoder(view1))
        z2 = self.projection_head(self.encoder(view2))
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(z.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class VICRegMedMNIST(nn.Module):
    def __init__(self, encoder_dim=512, projection_dim=2048,
                 lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0, pretrained=None):
        super().__init__()
        self.encoder = get_resnet18_medmnist(pretrained=pretrained, encoder_dim=encoder_dim)
        self.projection_head = VICRegProjectionHead(encoder_dim, 2048, projection_dim)
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(self, view1, view2):
        z1 = self.projection_head(self.encoder(view1))
        z2 = self.projection_head(self.encoder(view2))

        inv_loss = F.mse_loss(z1, z2)
        var_loss = self.variance_loss(z1) + self.variance_loss(z2)
        cov_loss = self.covariance_loss(z1) + self.covariance_loss(z2)

        loss = self.lambda_inv * inv_loss + self.lambda_var * var_loss + self.lambda_cov * cov_loss
        return loss

    def variance_loss(self, z):
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std))

    def covariance_loss(self, z):
        batch_size, dim = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (batch_size - 1)
        off_diagonal = cov.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
        return off_diagonal.pow(2).sum() / dim


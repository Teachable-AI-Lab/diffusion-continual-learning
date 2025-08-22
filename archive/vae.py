import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # 28→28
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 28→13  ⟵ switched to 4×4
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 13→13
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 13→5   ⟵ switched to 4×4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 5→5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 5→1    ⟵ switched to 4×4
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # Flatten the output to 128 channels
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # 1→5    ⟵ 4×4 upsample
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 5→5
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 5→13   ⟵ 4×4 upsample
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 13→13
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 13→28  ⟵ 4×4 upsample (needs output_padding=1 to land on 28)
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 28→28
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
    
class VAE(nn.Module):
    def __init__(self, num_classes=0):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.n_classes = num_classes
        self.decoder = Decoder()
        self.fc_mu = nn.Linear(128 + self.n_classes, 20)  # 128 channels, 1x1 feature map
        self.fc_logvar = nn.Linear(128 + self.n_classes, 20)
        self.decoder_fc = nn.Sequential(
            nn.Linear(20 + self.n_classes, 128),  # 20 latent dimensions
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 1, 1))  # Unflatten to (B, 128, 1, 1)
        )
        self.label_embedding = nn.Embedding(num_classes, num_classes) if num_classes > 0 else None

    def encode(self, x, y=None):
        x = self.encoder(x) # (B, 128)
        if y is not None:
            # y_one_hot = F.one_hot(y, num_classes=self.n_classes).float().to(x.device)
            y_embed = self.label_embedding(y)
            x = torch.cat((x, y_embed), dim=1)
        # x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y=None):
        if y is not None:
            # y_one_hot = F.one_hot(y, num_classes=self.n_classes).float().to(z.device)
            y_embed = self.label_embedding(y)
            z = torch.cat((z, y_embed), dim=1)
        z = self.decoder_fc(z)
        return self.decoder(z)

    def forward(self, x, y=None):
        # x = x.view(x.size(0), -1)  # Flatten
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z, y)
        # vae loss from scratch
        reconstruction_loss = F.mse_loss(x_reconstructed, x) * x.view(x.size(0), -1).size(1)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # print(f"Reconstruction Loss: {reconstruction_loss.item()}, KL Loss: {kl_loss.item()}")
        loss = reconstruction_loss + kl_loss
        return loss, x_reconstructed, mu, logvar
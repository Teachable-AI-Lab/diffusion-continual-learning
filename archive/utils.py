import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, IterableDataset, Dataset, Subset
from tqdm import tqdm
import torch_fidelity


def get_cl_dataset(name='mnist', batch_size=64, normalize=True, greyscale=False):
    if name.lower() == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif name.lower() == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif name.lower() == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if greyscale:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        train_dataset = None
        test_dataset = None

     # how many classes per group?
    group_size = 2
    n_classes  = 10
    n_groups   = n_classes // group_size  # == 5

    # return a dictionary: class_id: dataloader
    # train_indices_per_class = {i: [] for i in range(10)}
    train_indices_per_group = {g: [] for g in range(n_groups)}
    print("Building DataLoaders for each class in train dataset...")
    # for idx, (_, label) in enumerate(tqdm(train_dataset)):
        # train_indices_per_class[label].append(idx)
    for idx, (_, label) in enumerate(tqdm(train_dataset)):
        g = label // group_size
        train_indices_per_group[g].append(idx)

    # 3) Build one DataLoader per class
    train_loaders = {}
    # for class_id, indices in sorted(train_indices_per_class.items()):
    for g, indices in sorted(train_indices_per_group.items()):
        subset = Subset(train_dataset, indices)
        # train_loaders[class_id] = DataLoader(
        train_loaders[g] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,    # adjust as needed
            pin_memory=True
        )

    # test_indices_per_class = {i: [] for i in range(10)}
    test_indices_per_group = {g: [] for g in range(n_groups)}
    print("Building DataLoaders for each class in MNIST test dataset...")
    # for idx, (_, label) in enumerate(tqdm(test_dataset)):
        # test_indices_per_class[label].append(idx)
    for idx, (_, label) in enumerate(tqdm(test_dataset)):
        g = label // group_size
        test_indices_per_group[g].append(idx)
    # 3) Build one DataLoader per class
    test_loaders = {}
    # for class_id, indices in sorted(test_indices_per_class.items()):
    for g, indices in sorted(test_indices_per_group.items()):
        subset = Subset(test_dataset, indices)
        # test_loaders[class_id] = DataLoader(
        test_loaders[g] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,    # adjust as needed
            pin_memory=True
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loaders, test_loaders, train_loader, test_loader

def compute_fid(generated_images, real_images):
    fid = torch_fidelity.calculate_metrics(
        input1=generated_images,
        input2=real_images,
        cuda=True,
        isc=False,
        kid=False,
        fid=True,
        verbose=False,
    )
    return fid['frechet_inception_distance']

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        # Compute Fisher Information
        model.eval()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            x.requires_grad_(True)
            # _, recon, mu, logvar = model(x, y)
            # loss = vae_loss(recon, x, mu, logvar)
            pred_noise, noise = model(x, y=y)
            loss = F.mse_loss(pred_noise, noise)
            # loss, x_reconstructed, mu, logvar = model(x)

            model.zero_grad()
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.pow(2) * len(x)
        
        # Average over all batches
        for n in self.fisher:
            self.fisher[n] /= len(dataloader.dataset)
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss

class EWC_Rank1:
    def __init__(self, model, dataloader, device):
        self.model       = model
        self.dataloader  = dataloader
        self.device      = device

        # save old params
        self.params_old = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        # initialize running sum of score
        self.score = {n: torch.zeros_like(p) for n, p in self.params_old.items()}

        model.eval()
        N = len(dataloader.dataset)
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            x.requires_grad_(True)

            pred_noise, noise = model(x, y=y)
            loss = F.mse_loss(pred_noise, noise)

            model.zero_grad()
            loss.backward()

            # accumulate negative gradient as score
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # note the minus sign: score = -∇θ ℓ
                    self.score[n] += (-p.grad) * len(x)

        # average over dataset
        for n in self.score:
            self.score[n] /= N

    def penalty(self, model):
        # compute Δθ = θ - θ_old
        inner = 0.0
        for n, p in model.named_parameters():
            Δ = p - self.params_old[n]
            inner += (self.score[n] * Δ).sum()
        # rank-1 penalty = (s^T Δθ)^2
        return inner.pow(2)

    
class EWCDDPM:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        # Compute Fisher Information
        model.eval()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            x.requires_grad_(True)
            # _, recon, mu, logvar = model(x, y)
            # loss = vae_loss(recon, x, mu, logvar)
            pred_noise, noise = model(x, y=y)
            loss = F.mse_loss(pred_noise, noise)
            # loss, x_reconstructed, mu, logvar = model(x)

            model.zero_grad()
            loss.backward()
            # grad = torch.autograd.grad(
            #     outputs=loss,
            #     inputs=model.parameters(),
            #     create_graph=False,
            #     retain_graph=False,
            # )
            
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # eigenvector = -p.grad
                    # normalize the eigenvector
                    # eigenvector = eigenvector / eigenvector.norm()
                    eigenvalue = 1#eigenvector.norm().item()
                    self.fisher[n] += p.grad.pow(2)#eigenvector* eigenvalue
        
        # Average over all batches
        for n in self.fisher:
            self.fisher[n] /= len(dataloader.dataset)
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            loss += self.fisher[n] * (p - self.params[n]).pow(2).sum()
        return loss
    
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            # nn.Linear(512, 400),
            # nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            # nn.Linear(400, 512),
            # nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()  # Output layer with Sigmoid activation for pixel values
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = x.view(-1, 784)
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
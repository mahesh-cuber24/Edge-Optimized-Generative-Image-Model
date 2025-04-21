import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from models.gan import Generator, Discriminator
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
lr = 0.0002
epochs = 50
batch_size = 128

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = Generator(z_dim).to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()
g_opt = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_opt = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    for i, (real, _) in enumerate(dataloader):
        real = real.to(device)
        batch = real.size(0)
        noise = torch.randn(batch, z_dim, 1, 1, device=device)
        fake = G(noise)

        # Train Discriminator
        D_real = D(real).view(-1)
        D_fake = D(fake.detach()).view(-1)
        D_loss = loss_fn(D_real, torch.ones_like(D_real)) + loss_fn(D_fake, torch.zeros_like(D_fake))

        d_opt.zero_grad()
        D_loss.backward()
        d_opt.step()

        # Train Generator
        G_fake = D(fake).view(-1)
        G_loss = loss_fn(G_fake, torch.ones_like(G_fake))

        g_opt.zero_grad()
        G_loss.backward()
        g_opt.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {D_loss:.4f} | G_loss: {G_loss:.4f}")

    if (epoch+1) % 10 == 0:
        torchvision.utils.save_image(fake[:25], f"outputs/fake_epoch_{epoch+1}.png", nrow=5, normalize=True)

torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")

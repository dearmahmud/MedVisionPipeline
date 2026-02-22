import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.dataset import get_dataloaders
from models.resnet_model import ResNetPneumonia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _ = get_dataloaders()

model = ResNetPneumonia(pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "resnet_pneumonia.pth")
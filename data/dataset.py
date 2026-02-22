from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train = PneumoniaMNIST(split='train', transform=train_transform, download=True)
    val = PneumoniaMNIST(split='val', transform=test_transform, download=True)
    test = PneumoniaMNIST(split='test', transform=test_transform, download=True)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size)
    )
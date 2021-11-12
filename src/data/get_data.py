from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# NOTE: MedMNIST images are small, can use 28x28
# TODO: add args parsing later
def get_loaders(batch_size=64, download=True):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        # normalization is kind of random here but ok-ish
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = PneumoniaMNIST(split="train", transform=tfm, download=download)
    val_ds = PneumoniaMNIST(split="val", transform=tfm, download=download)
    test_ds = PneumoniaMNIST(split="test", transform=tfm, download=download)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
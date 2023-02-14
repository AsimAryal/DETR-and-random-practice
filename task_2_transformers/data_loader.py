from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TransformerDataLoader:
    def get_train_test_loaders(
        self,
        transform: transforms.Compose,
        batch_size: int,
        train_dir: str = "C:/Users/asima/PycharmProjects/AAGI_Prac/data_intel/train/",
        test_dir: str = "C:/Users/asima/PycharmProjects/AAGI_Prac/data_intel/val/",
    ):
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        class_names = train_data.classes

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader, class_names

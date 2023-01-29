import torch.utils.data
from data_loader import CustomDataLoader
from torch.utils.data import DataLoader

if __name__ == "__main__":
    batch_size = 32
    dataset = CustomDataLoader(100)
    print(f"Size of dataset: {dataset.__len__()}")
    train_set, test_set = torch.utils.data.random_split(dataset, [80, 20])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    print(f"images per batch: {train_loader.__len__()}")
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    new_set = next(CustomDataLoader(1000))
    print("Size of new set: ", new_set.__len__())

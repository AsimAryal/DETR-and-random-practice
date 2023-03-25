import torch.utils.data
from data_loader import TerrainDataLoader
from torch.utils.data import DataLoader

if __name__ == "__main__":
    batch_size = 32
    dataset = TerrainDataLoader(1500)
    print(f"Size of dataset: {dataset.__len__()}")
    split_ratio = 0.8
    train_size = int(split_ratio * dataset.__len__())
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    print(f"images per batch: {train_loader.__len__()}")
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    new_set = next(TerrainDataLoader(1000))
    print("Size of new set: ", new_set.__len__())

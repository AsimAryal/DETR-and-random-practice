import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CustomDataLoader(Dataset):
    def __init__(
        self,
        size: int,
        root_dir: Path = Path("C:/Users/asima/PycharmProjects/AAGI_Prac/data_terrain/"),
    ):
        self.classes = ["desert", "green", "cloudy", "water"]
        self.list_of_filenames = []

        self.size = size
        self.root_dir = root_dir
        self.folder_sizes = self.check_and_get_folder_sizes(size, self.classes)
        for folder in self.classes:
            size_of_folder = self.folder_sizes[folder]
            size_to_get = size_of_folder if size_of_folder < size else size
            self.list_of_filenames.append(self.get_all_filenames(folder, size_to_get))
        self.filenames = np.transpose(
            pd.DataFrame.from_dict(
                dict(zip(self.classes, self.list_of_filenames)), orient="index"
            )
        )

    def get_all_filenames(self, folder_name: str, number_of_files: int) -> list[str]:
        return random.sample(self.get_filenames(folder_name), number_of_files)

    def get_filenames(self, folder_name: str) -> list[str]:
        return next(os.walk(Path(self.root_dir, folder_name)), (None, None, []))[2]

    def get_folder_size(self, folder_name: str) -> int:
        return len(self.get_filenames(folder_name))

    def check_and_get_folder_sizes(self, size_to_check: int, folders: list[str]):
        sizes = [self.get_folder_size(folder_name) for folder_name in folders]
        if size_to_check > max(sizes):
            raise ValueError("Minimise the sample size to less than ", max(sizes))
        return dict(zip(folders, sizes))

    @staticmethod
    def export_to_csv(filenames_df: pd.DataFrame):
        filenames_df.to_csv("filenames.csv", sep=",")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        pass

    def __iter__(self):
        return self

    def __next__(
        self,
    ):
        return CustomDataLoader(self.size)


#
# print(
#     (
#         CustomDataLoader(
#             Path("C:/Users/asima/PycharmProjects/AAGI_Prac/data_terrain/"), 50
#         )
#     )
# )

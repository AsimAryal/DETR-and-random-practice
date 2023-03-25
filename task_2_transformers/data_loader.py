from torch.utils.data import Dataset
from task_2_transformers.data_utils import get_coco_image_annotation, get_coco_objects


class TransformerDataSet(Dataset):
    def __init__(
        self,
        train_path="C:/Users/asima/PycharmProjects/AAGI_Prac/data_turbines/annotations/_train_annotations.coco.json",
        val_path="C:/Users/asima/PycharmProjects/AAGI_Prac/data_turbines/annotations/_val_annotations.coco.json",
    ):

        self.train_images, self.train_annotations = get_coco_image_annotation(
            get_coco_objects(train_path)
        )
        self.val_images, self.val_annotations = get_coco_image_annotation(
            get_coco_objects(val_path)
        )

    def __len__(self):
        return len(self.train_annotations)

    def __getitem__(self, item):
        pass

import torch
import torchvision
from torch import nn
from torchinfo import summary
from torchvision import transforms

from task_2_transformers.data_loader import TransformerDataSet


class SetupTraining:
    def __init__(self):
        self.img_size = 224
        self.batch_size = 32
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )

    def get_data(self):
        return TransformerDataSet().get_train_test_loaders(
            transform=self.image_transforms, batch_size=self.batch_size
        )

    def get_training_image_batch(self):
        train_loader, test_loader, class_names = self.get_data()
        image_batch, label_batch = next(iter(train_loader))
        return image_batch, label_batch

    def get_random_image_tensor(
        self, batch_size=32, colour_channels=3, height=224, width=224
    ):
        return torch.randn(batch_size, colour_channels, height, width)


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector."""

    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ):
        """

        :param in_channels: Number of color channels for the input images. Defaults to 3
        :param patch_size: Size of patches to convert input image into. Defaults to 16
        :param embedding_dim: Size of embedding to turn image into. Defaults to 768
        """
        super().__init__()

        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        # Create a layer to flatten the patch feature maps into a single dimension
        # only flatten the feature map dimensions into a single vector
        self.flatten = nn.Flatten(
            start_dim=2,
            end_dim=3,
        )

    def forward(self, x):
        # assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, (
            f"Input image size must be divisible by patch size, +"
            f"image shape: {image_resolution}, patch size: {self.patch_size}"
        )
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # Make sure the output shape has the right order
        # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
        return x_flattened.permute(0, 2, 1)

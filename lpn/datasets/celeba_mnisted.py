from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CelebA


class CelebADataset(Dataset):
    def __init__(self, root, split, image_size):
        # original size: 218*178
        transform = transforms.Compose(
            [
                transforms.CenterCrop(image_size),
                transforms.Resize(
                    (28,28), ## change tupple size to fit MNIST network
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.Grayscale(num_output_channels=1), ## Change to make grayscale to fit MNIST network
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self.dataset = CelebA(
            root=root, split=split, transform=transform, download=False
        )

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        return {"image": img, "label": lb}

    def __len__(self):
        return len(self.dataset)


LPNDataset = CelebADataset

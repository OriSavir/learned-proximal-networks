from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class BSDDataset(Dataset):
    def __init__(self, root, split="train", image_size=128):
        self.image_dir = os.path.join(root, split)
        self.image_paths = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith(".jpg")
        ])

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ])

        self.dataset = [{"fn": os.path.basename(p)} for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # shape: (H, W, 3)
        image = self.transform(image)  # shape: (3, 128, 128)
        return {"image": image, "fn": self.dataset[idx]["fn"]}

LPNDataset = BSDDataset
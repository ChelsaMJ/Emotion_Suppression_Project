# test_loader.py

from src.data_loader import CASME2Dataset
from torchvision import transforms

dataset = CASME2Dataset(
    image_root="data/CASME II/Cropped",
    label_excel_path="data/CASME II/CASME2-coding-20140508.xlsx",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)

print(f"Loaded {len(dataset)} image samples")

img, label = dataset[0]
print(f"Image shape: {img.shape}")
print(f"Label: {label}")
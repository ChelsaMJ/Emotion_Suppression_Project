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

from src.train import train_model

train_model(
    image_root="data/CASME II/Cropped",
    label_excel="data/CASME II/CASME2-coding-20140508.xlsx",
    num_classes=6,
    num_epochs=5  # increase later
)

print(f"Loaded {len(dataset)} samples")
img, label = dataset[0]
print(img.shape, label)
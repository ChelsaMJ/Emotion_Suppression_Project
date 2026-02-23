from src.data_loader import CASME2Dataset
from src.train import train_model
from torchvision import transforms

# Path setup
image_root = "data/CASME II/Cropped"
label_excel = "data/CASME II/CASME2-coding-20140508.xlsx"

# Load dataset and preview one sample
dataset = CASME2Dataset(
    image_root=image_root,
    label_excel_path=label_excel,
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
)

print(f"----Loaded {len(dataset)} samples")
img, label = dataset[0]
print("Sample image shape:", img.shape)
print("Sample label:", label)

# Train the model
train_model(
    image_root=image_root,
    label_excel=label_excel,
    num_classes=7,        
    num_epochs=30          # Increase for stronger learning
)
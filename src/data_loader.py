import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Emotion label mapping
EMOTION_MAP = {
    "happiness": 0,
    "disgust": 1,
    "surprise": 2,
    "repression": 3,
    "others": 4,
    "fear": 5,
    "sadness": 6
}

class CASME2Dataset(Dataset):
    def __init__(self, image_root, label_excel_path, transform=None):
        self.image_root = image_root
        self.label_df = pd.read_excel(label_excel_path)
        self.transform = transform if transform else transforms.ToTensor()
        self.samples = []

        self._prepare_dataset()

    def _prepare_dataset(self):
        for _, row in self.label_df.iterrows():
            sub = row['Subject']
            filename = row['Filename']
            emotion = str(row['Estimated Emotion']).strip().lower()

            if emotion not in EMOTION_MAP:
                continue

            label = EMOTION_MAP[emotion]
            folder = f"sub{sub:02d}/{filename}"
            folder_path = os.path.join(self.image_root, folder)

            if not os.path.exists(folder_path):
                continue

            for fname in os.listdir(folder_path):
                if fname.endswith(".jpg") or fname.endswith(".png"):
                    self.samples.append({
                        "img_path": os.path.join(folder_path, fname),
                        "label": label
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        img = self.transform(img)
        return img, sample["label"]
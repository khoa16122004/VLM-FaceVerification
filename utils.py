import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, type="same"):
        self.root_dir = root_dir
        self.type = type
        self.data = []
        
        type_dir = os.path.join(root_dir, type)
        for case_name in os.listdir(type_dir):
            for id in os.listdir(os.path.join(type_dir, case_name)):
                folder_id = os.path.join(type_dir, case_name, id)
                img1_path, img2_path = [os.path.join(folder_id, img_name) for img_name in os.listdir(folder_id)]
                self.data.append((type, case_name, (img1_path, img2_path)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, case_name, (img1_path, img2_path) = self.data[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        return label, case_name, (img1, img2), (img1_path, img2_path)
        


import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import json
import re

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
    
    
class LFW(Dataset):        
    def __init__(self, 
                 IMG_DIR: str,
                 PAIR_PATH: str,
                 transform=None,
                 ):
        
        with open(PAIR_PATH, "r") as f:
            f.readline()
            lines = [line.strip().split("\t") for line in f.readlines()]
         
        self.lines = lines
        self.IMG_DIR = IMG_DIR
        self.transform = transform
         
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        if len(line) == 3:
            first_iden_name, first_id, second_id = line
            second_iden_name = first_iden_name 
            label = 0           
        elif len(line) == 4:
            first_iden_name, first_id, second_iden_name, second_id = line
            label = 1
        
        first_name = f"{first_iden_name}_{first_id.zfill(4)}.jpg" 
        first_path = os.path.join(self.IMG_DIR, first_iden_name, first_name)        
        
        second_name = f"{second_iden_name}_{second_id.zfill(4)}.jpg"
        second_path =  os.path.join(self.IMG_DIR, second_iden_name, second_name)
        
        
        first_image = Image.open(first_path).convert("RGB")
        second_image = Image.open(second_path).convert("RGB")
        
        
        if self.transform:
            first_image = self.transform(first_image)
            second_image = self.transform(second_image) 

                
        return first_image, second_image, label
        
        
        


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_txt(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(str(content))

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def extract_answer(text, llm):
    system_prompt = (
        "Given a paragraph describing two facial images, determine whether they show the same person or not. "
        "Reply with only one word: 'same' or 'different'. "
        "If you are uncertain or the paragraph is ambiguous, choose 'same'."
    )
    prompt = text

    response = llm.text_to_text(system_prompt, prompt)[0].strip().lower()

    # Nếu phản hồi không phải là "same" hoặc "different", cố gắng trích xuất bằng regex
    while response not in ["same", "different"]:
        match = re.search(r'\b(same|different)\b', response)
        if match:
            response = match.group(1)
        else:
            # Gọi lại mô hình nếu vẫn không rõ ràng
            response = llm.text_to_text(system_prompt, prompt)[0].strip().lower()

    return response
    
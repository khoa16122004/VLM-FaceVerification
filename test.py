from face_verification import FaciVerification, DetectiveGame
from utils import CustomDataset
dataset = CustomDataset(root_dir=r"samples", 
                        type="different")

print(dataset[0])

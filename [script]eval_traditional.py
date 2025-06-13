import argparse
from utils import LFW, save_txt, save_json, ensure_dir
from traditional_arch import get_face_encoder
import torch
from PIL import Image
from torchvision import transforms  
import torch.nn.functional as F
from sklearn.metrics import f1_score

def main(args):
    model, img_size = get_face_encoder(args.model_name)

    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    ])
    dataset = LFW(
        IMG_DIR=args.img_dir,
        PAIR_PATH=args.pair_path,
        transform=transform,
    )
    
    outputs = []
    acc_0 = 0
    acc_1 = 0
    total_0 = 0
    total_1 = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img1, img2, label = dataset[i]
            img1_embedding = model(img1.unsqueeze(0).cuda())
            img2_embedding = model(img2.unsqueeze(0).cuda())
            
            img1_embedding = F.normalize(img1_embedding, p=2, dim=1)
            img2_embedding = F.normalize(img2_embedding, p=2, dim=1)
            
            sim = torch.abs(img1_embedding @ img2_embedding.T).item()
            print("sim: ", sim)
            
            pred = 0 if sim >= args.threshold else 1
            y_true.append(label)
            y_pred.append(pred)

            if label == 0:
                total_0 += 1
                if pred == 0:
                    acc_0 += 1
            elif label == 1:
                total_1 += 1
                if pred == 1:
                    acc_1 += 1

    acc_0 = acc_0 / total_0 if total_0 else 0
    acc_1 = acc_1 / total_1 if total_1 else 0
    overall_acc = (acc_0 * total_0 + acc_1 * total_1) / len(dataset)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy (Same):      {acc_0:.4f}")
    print(f"Accuracy (Different): {acc_1:.4f}")
    print(f"Overall Accuracy:     {overall_acc:.4f}")
    print(f"Macro F1 Score:       {macro_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="Path to image directory", default="lfw/images")
    parser.add_argument("--model_name", type=str, default="restnet_vggface")
    parser.add_argument("--pair_path", type=str, help="Path to LFW pair.txt file", default="lfw/pairs.txt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
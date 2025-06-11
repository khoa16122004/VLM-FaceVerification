import os
import json
import argparse
from face_verification import FaceVerification, DetectiveGame
from utils import LFW, save_txt, save_json, ensure_dir, extract_answer
from tqdm import tqdm
from llm_service import LlamaService
from sklearn.metrics import f1_score

def main(args):
    dataset = LFW(
        IMG_DIR=args.img_dir,
        PAIR_PATH=args.pair_path,
        transform=None,
    )

    with open(args.result_path, "r") as f:
        results = [line.strip().lower() for line in f.readlines()]

    assert len(results) == len(dataset), "Result file size does not match dataset size."

    acc_same = 0
    acc_diff = 0
    total_same = 0
    total_diff = 0

    y_true = []
    y_pred = []

    wrong_same_indexes = []
    wrong_diff_indexes = []

    for i in tqdm(range(len(dataset)), desc="Processing Samples"):
        img1, img2, label = dataset[i]
        output = results[i]

        y_true.append(label)
        pred = 0 if output == "same" else 1
        y_pred.append(pred)

        if label == 0:
            total_same += 1
            if output == "same":
                acc_same += 1
            else:
                wrong_same_indexes.append(str(i))
        elif label == 1:
            total_diff += 1
            if output == "different":
                acc_diff += 1
            else:
                wrong_diff_indexes.append(str(i))

    same_acc = acc_same / total_same if total_same else 0
    diff_acc = acc_diff / total_diff if total_diff else 0
    overall_acc = (acc_same + acc_diff) / len(dataset)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy (Same):      {same_acc:.4f}")
    print(f"Accuracy (Different): {diff_acc:.4f}")
    print(f"Overall Accuracy:     {overall_acc:.4f}")
    print(f"Macro F1 Score:       {macro_f1:.4f}")
    print()

    # Save wrong cases
    save_txt("wrong_same.txt", [str(i) for i in wrong_same_indexes])
    save_txt("wrong_diff.txt", [str(i) for i in wrong_diff_indexes])
    print("Saved wrong indices to wrong_same.txt and wrong_diff.txt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="Path to image directory", default="lfw/images")
    parser.add_argument("--pair_path", type=str, help="Path to LFW pair.txt file", default="lfw/pairs.txt")
    parser.add_argument("--result_path", type=str, help="Path to file containing model outputs", required=True)

    args = parser.parse_args()
    main(args)

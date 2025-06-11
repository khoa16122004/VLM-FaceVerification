import os
import json
import argparse
from face_verification import FaceVerification, DetectiveGame
from utils import LFW, save_txt, save_json, ensure_dir, extract_answer
from tqdm import tqdm
from llm_service import LlamaService

def main(args):
    # Prepare dataset
    dataset = LFW(
        IMG_DIR=args.img_dir,
        PAIR_PATH=args.pair_path,
        transform=None,
    )
    
    # extract llm
    llm = LlamaService("Llama-7b")
    
    output_path = args.input_dir + ".txt"
    with open(output_path, "w") as f:
        

        for i in tqdm(range(len(dataset)), desc="Processing Samples"):
            sample_dir = os.path.join(args.input_dir, f"sample_{i}")
            with open(os.path.join(sample_dir, "decision.txt"), "r") as decision_file:
                decision = decision_file.read().strip()
                print("Decision:", decision)
                output = extract_answer(decision, llm)
                print(output)
                f.writr(output + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="Path to image directory", default="lfw/images")
    parser.add_argument("--pair_path", type=str, help="Path to LFW pair.txt file", default="lfw/pairs.txt")

    parser.add_argument("--input_dir", type=str, help="Path to input directory containing samples", required=True)

    args = parser.parse_args()
    main(args)
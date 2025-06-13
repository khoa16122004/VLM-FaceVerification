import os
import json
import argparse
from face_verification import FaceVerification, DetectiveGame
from utils import LFW, save_txt, save_json, ensure_dir
from tqdm import tqdm

def main(args):
    # Prepare dataset
    dataset = LFW(
        IMG_DIR=args.img_dir,
        PAIR_PATH=args.pair_path,
        transform=None,
    )

    # Prepare controller
    vlm_info = (args.pretrained_lvlm, args.model_name_lvlm)
    llm_info = (args.llm_model,) if args.llm_model else None

    controller = FaceVerification(vlm_model=vlm_info, llm_model=llm_info)

    # Output dir
    output_root = f"{args.prefix}_seed={args.seed}_direct={args.direct_return}_vlm={args.pretrained_lvlm}"
    ensure_dir(output_root)

    for i in tqdm(range(args.start_index, len(dataset)), desc="Processing Samples"):
        img1, img2, label = dataset[i]

        response = controller.simple_answer(img1, img2, direct_return=args.direct_return)
        
        sample_dir = os.path.join(output_root, f"sample_{i}")
        ensure_dir(sample_dir)

        save_txt(os.path.join(sample_dir, "decision.txt"), response)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="Path to image directory", default="lfw/images")
    parser.add_argument("--pair_path", type=str, help="Path to LFW pair.txt file", default="lfw/pairs.txt")
    parser.add_argument("--pretrained_lvlm", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name_lvlm", type=str, default="llava_qwen")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--direct_return", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()
    main(args)
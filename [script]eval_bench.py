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

    if args.controller == "traditional":
        controller = FaceVerification(vlm_model=vlm_info, llm_model=llm_info)
    elif args.controller == "detective":
        controller = DetectiveGame(vlm_model=vlm_info, llm_model=llm_info)
    else:
        raise ValueError("Invalid controller type")

    # Output dir
    output_root = "controller={args.controller}_vlm={args.pretrained_lvlm}_llm={args.llm_model}"
    ensure_dir(output_root)

    for i in tqdm(range(len(dataset)), desc="Processing Samples"):
        img1, img2, label = dataset[i]

        final_decision, all_qas, selection_qas, summary = controller.sampling_answer(img1, img2)
        
        sample_dir = os.path.join(output_root, f"sample_{i}")
        ensure_dir(sample_dir)

        save_txt(os.path.join(sample_dir, "label.txt"), str(label))
        save_txt(os.path.join(sample_dir, "decision.txt"), final_decision)
        save_json(os.path.join(sample_dir, "all_questions.json"), all_qas)
        save_json(os.path.join(sample_dir, "selections.json"), selection_qas)
        save_txt(os.path.join(sample_dir, "summary.txt"), summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="Path to image directory", default="lfw/images")
    parser.add_argument("--pair_path", type=str, help="Path to LFW pair.txt file", default="lfw/pairs.txt")
    parser.add_argument("--pretrained_lvlm", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name_lvlm", type=str, default="llava_qwen")
    parser.add_argument("--llm_model", type=str, default="Llama-7b")
    parser.add_argument("--controller", choices=["traditional", "detective"], default="traditional")

    args = parser.parse_args()
    main(args)
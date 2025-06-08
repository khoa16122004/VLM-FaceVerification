from face_verification import FaceVerification, DetectiveGame
from utils import CustomDataset
from get_arch import init_lvlm_model
from llm_service import QwenService, GPTService, LlamaService

# init_dataset
dataset = CustomDataset(root_dir=r"samples", 
                        type="different")

# init model
lvlm_model, image_token, special_token =init_lvlm_model(pretained="llava-next-interleave-qwen-7b", 
                                                        model_name="llava_qwen")

llm_model = LlamaService(model_name="Llama-7b")

for (label, case_name, (img1, img2), (img1_path, img2_path)) in dataset:
    # init controller
    traditional_controller = FaceVerification(vlm_model=lvlm_model, 
                                              image_token=image_token, 
                                              llm_model=llm_model)
    # simple answer
    result = traditional_controller.simple_answer(img1, img2)
    print("Simple answer: ", result)
    
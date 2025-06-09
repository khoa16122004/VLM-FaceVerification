from face_verification import FaceVerification, DetectiveGame
from utils import CustomDataset
from get_arch import init_lvlm_model
from llm_service import QwenService, GPTService, LlamaService


# init_dataset
dataset = CustomDataset(root_dir=r"samples", 
                        type="different")

# init model\
pretrained_lvlm = "llava-next-interleave-qwen-7b"
model_name_lvlm = "llava_qwen"
vlm_model = (pretrained_lvlm, model_name_lvlm)

llm_model = ("Llama-7b", )
# llm_model = LlamaService(model_name="llama-3-8b")
# llm_model=None


# ciontroller
traditional_controller = FaceVerification(vlm_model=vlm_model, 
                                            llm_model=llm_model)

for (label, case_name, (img1, img2), (img1_path, img2_path)) in dataset:

    print("image 1: ", img1_path)
    print("image 2: ", img2_path)
    
    
    
    # simple answer
    # result = traditional_controller.simple_answer(img1, img2)
    # print("Simple answer: ", result)
    
    # # explain answer
    # result = traditional_controller.simple_answer(img1, img2, direct_return=0)
    # print("Explain answer: ", result)
    final_decision, all_question_responses, selection_responses, summarized_responses = traditional_controller.sampling_answer(img1, img2)
    print("Final decision: ", final_decision)
    input("Press Enter to continue...")
    # # sampling answer
    # final_decision, all_question_responses, selection_responses = traditional_controller.sampling_answer(img1, img2)
    # print("Sampling decision: ", final_decision)
    # break    
    
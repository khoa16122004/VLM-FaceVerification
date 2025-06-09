import torch
from llm_service import QwenService, GPTService, LlamaService
from get_arch import init_lvlm_model

class FaceVerification:
    def __init__(self, 
                 vlm_model, # (pretrained, model_name) 
                 llm_model=None): # (model_name,)
        
        self.vlm_model, self.image_token, self.special_token = init_lvlm_model(pretrained=vlm_model[0], 
                                                                              model_name=vlm_model[1])
        if llm_model:
            self.llm_model = LlamaService(model_name=llm_model[0])
        
    @torch.no_grad()
    def simple_answer(self, img1, img2, direct_return=1):
        if direct_return == 1:
            prompt = (
                f"Analyze the two facial images and determine if they belong to the same person. "
                f"Focus especially on biometric regions such as eyes, nose, mouth, and jawline. "
                f"Respond with only one word: 'Same' or 'Different'. Do not provide any explanation. "
                f"{self.image_token} {self.image_token}"
            )
        else:
            prompt = (
                f"Analyze the two facial images and decide whether they belong to the same person. "
                f"Focus especially on biometric regions such as eyes, nose, mouth, and jawline. "
                f"First, explain your reasoning in detail based on these regions. Then, clearly state your final conclusion as either 'Same' or 'Different'. "
                f"{self.image_token} {self.image_token}"
            )

        response = self.vlm_model.inference(
            qs=prompt,
            img_files=[img1, img2],
            num_return_sequences=1,
            do_sample=False,
            temperature=0
        )[0].replace("\n", "")

        return response


    
    @torch.no_grad()
    def sampling_answer(self, img1, img2, num_samples=3, temparature=0.8):
        questions = [
            "Do the eyes of the two individuals have similar size and shape?",
            "Is there a noticeable difference in the nose length and width between the two individuals?",
            "Are the mouths of the two individuals similar in terms of lip thickness and symmetry?",
            "Do the facial structures, such as the jaw and chin, appear similar?",
            "Do the individuals have similar eyebrow shapes, density, or gaps between brows?"
        ]
        selection_voting = (
            "You will receive multiple brief opinions on a binary question.",
            "Treat these as votes and determine the majority viewpoint.",
            "Summarize the overall consensus in a short sentence, focusing only on the main idea."
        )
        
        conclusion_summarize_prompt = (
            "Summarize the following multiple responses into a concise consensus statement:"
        )
        
        conclusion_prompt_template = (
            "Given the responses describing facial features in two images, treat each response as a 'vote' indicating whether the images depict the same person or different individuals.\n"
            "Assign greater weight to responses that mention differences in key biometric features (e.g., eye shape, jawline, nose structure).\n"
            "Based on the overall weighted vote, determine whether the images likely show the same person or not.\n"
            "Here are the responses:\n"
            "{responses}"
        )
        
        all_question_responses = []
        selection_responses = []
        
        for question in questions:
            prompt = f"{question} {self.image_token} {self.image_token}"
            outputs = self.vlm_model.inference(
                qs=prompt,
                img_files=[img1, img2],
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=temparature
            )
            all_question_responses.append(outputs)
            
            prompt_for_llm = f"Question: {question}\nResponses:\n" + "\n".join(f"- {o}" for o in outputs)
            output = self.llm_model.text_to_text(
                system_prompt=selection_voting,
                prompt=prompt_for_llm
            )
            selection_responses.append(output)
        
        # Bước tóm tắt selection_responses bằng LLM trước
        responses_text = "\n".join(f"- {resp}" for resp in selection_responses)
        summarized_responses = self.llm_model.text_to_text(
            system_prompt=conclusion_summarize_prompt,  # hoặc bạn có thể để prompt tùy ý cho LLM tóm tắt
            prompt=responses_text
        )
        
        # Rồi dùng kết quả tóm tắt này để làm prompt cho VLM kết luận
        final_prompt = conclusion_prompt_template.format(responses=summarized_responses)
        
        final_decision = self.vlm_model.inference(
            qs=final_prompt,
            img_files=[img1, img2],
            num_return_sequences=1,
            do_sample=False,
            temperature=0
        )[0].replace("\n", "")
        
        return final_decision, all_question_responses, selection_responses, summarized_responses



    

class DetectiveGame:
    def __init__(self, lvlm_model,
                 llm,
                 max_rounds=3,
                 image_token="<image>"):
        self.lvlm_model = lvlm_model
        self.llm = llm
        self.image_token = image_token
        self.max_rounds = max_rounds

        self.initial_instruction = """
🎮 DETECTIVE CHALLENGE: Guess if two faces are the same person by asking the FEWEST yes/no questions!

🕵️ Your Mission: You’re a master detective who cannot see the images. Two witnesses each have one different image. You ask them the same yes/no question. Each witness answers yes or no based on their own image.

🎯 GAME RULES:
- Ask one yes/no question to both witnesses.
- Compare their yes/no answers to decide.
- When confident, respond with "None" to finish.
- Do NOT ask comparative questions since each witness sees only their own image.
- Questions must be clear and specific enough to allow comparison.
⚠️ Important:
Example question: "Is the person in the image male?" (yes/no)
Avoid open-ended or descriptive questions.
"""
    
    def play(self, img1, img2):
        history = []
        chat_context = [(
            self.initial_instruction + "\nStart with your first question.",
            "Understood. Here's my first question:\nIs the person in the image male?"
        )]
        round_counting = 0
        for round_idx in range(self.max_rounds):
            current_question = chat_context[-1][1]

            print(f"\n🎮 Game Round {round_idx + 1}")
            print(f"🕵️ Detective Question: {current_question.strip()}")

            answer_1 = self.lvlm_model.inference(
                "Just answer yes or no.\n" + current_question + self.image_token,
                [img1],
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                reload=False
            )[0]

            answer_2 = self.lvlm_model.inference(
                "Just answer yes or no.\n" + current_question + self.image_token,
                [img2],
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                reload=False
            )[0]

            print(f"👤 Witness #1: {answer_1.strip()}")
            print(f"👤 Witness #2: {answer_2.strip()}")

            history.append((current_question, answer_1, answer_2))

            if "none" in current_question.lower():
                print("\n🎯 GAME OVER! Detective has reached a conclusion!")
                break

            # Format conversation history for LLM context
            chat_context.append((current_question, f"Witness 1: {answer_1}\nWitness 2: {answer_2}"))

            # Get next detective question from LLM
            next_question = self.llm_model.chat(chat_context, "What is your next yes/no question? If you're confident, say 'None'.")
            chat_context.append(("What is your next yes/no question? If you're confident, say 'None'.", next_question))

            round_counting += 1
            
        return history, round_counting
    

        
        
if __name__ == "main":
    
    
    
    
    lvlm_model, image_token, special_token =init_lvlm_model(pretained="llava-next-interleave-qwen-7b", 
                                                            model_name="llava_qwen")
    
    llm_model = LlamaService(model_name="Llama-7b")
    

    # direct return
    traditional_controller = FaciVerification(vlm_model=lvlm_model, 
                                              image_token=image_token, 
                                              llm_model=llm_model)
    
    
    
    
    
           
        

        
         
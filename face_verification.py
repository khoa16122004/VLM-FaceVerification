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
            "Do the two individuals appear to be of the same gender?",
            "Is there a noticeable difference in skin tone or complexion between the two individuals?",
            "Do the individuals have similar hair color, length, or style?",
            "Do they appear to be in the same age group based on facial features such as wrinkles or skin?",
            "Are there any distinctive facial marks or features (e.g., freckles, moles, scars) that set the individuals apart?"
        ]
        
        selection_voting = (
            "You will receive multiple brief opinions on a binary question.\n"
            "Treat these as votes and determine the majority viewpoint.\n"
            "Summarize the overall consensus in a short sentence, focusing only on the main idea."
        )
        
        conclusion_summarize_prompt = (
            "Summarize the following multiple responses into a concise statement that reflects the majority opinion:"
        )

        conclusion_prompt_template = (
            "You are given two face images and a summary of expert opinions comparing their biometric features, "
            "Based on this SUMMARY and the VISUAL content of the two images, provide a conclusion about whether they likely the same person or not"
            "Images: {img_token1} and {img_token2}\n\n"
            "Summary of expert responses:\n"
            "{responses}\n\n"
            "Your conclusion:"
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

        # Bước tóm tắt selection_responses bằng LLM
        responses_text = "\n".join(f"- {resp}" for resp in selection_responses)
        summarized_responses = self.llm_model.text_to_text(
            system_prompt=conclusion_summarize_prompt,
            prompt=responses_text
        )

        # Tạo prompt kết luận
        final_prompt = conclusion_prompt_template.format(
            responses=summarized_responses,
            img_token1=self.image_token,
            img_token2=self.image_token
        )

        final_decision = self.vlm_model.inference(
            qs=final_prompt,
            img_files=[img1, img2],
            num_return_sequences=1,
            do_sample=False,
            temperature=0
        )[0].replace("\n", "")

        return final_decision, all_question_responses, selection_responses, summarized_responses



class DetectiveGame:
    def __init__(self, 
                 vlm_model,  # (pretrained, model_name)
                 llm_model,  # (model_name,)
                 max_rounds=5):
        self.vlm_model, self.image_token, self.special_token = init_lvlm_model(
            pretrained=vlm_model[0], 
            model_name=vlm_model[1]
        )

        self.llm_model = LlamaService(model_name=llm_model[0])
        self.max_rounds = max_rounds

        self.instruction_prompt = (
            "🎮 DETECTIVE CHALLENGE: Determine if two face images belong to the same person by asking YES/NO questions.\n\n"
            "🕵️ You are a detective who **cannot see the images**. Each witness sees **only one image** and cannot see the other's image.\n"
            "You must ask **the same yes/no question** to both witnesses, based solely on the face they see.\n\n"
            "🎯 RULES:\n"
            "- Ask only **clear, objective yes/no questions**.\n"
            "- Focus on **biometric features** visible in a single image, such as:\n"
            "  • Skin tone and color\n"
            "  • Hair style, color, or facial hair\n"
            "  • Facial structure and shapes (eyes, nose, lips, jawline)\n"
            "  • Ethnicity-related traits\n"
            "- Avoid subjective, comparative, or ambiguous questions.\n"
            "- When confident, reply with **'None'** to stop.\n"
            "- ⚠️ Only output the next yes/no question. Do not add explanations or other text."
        )


        self.summarize_prompt = (
            "Summarize briefly how similar or different the witnesses' yes/no answers were. "
            "Focus on whether their responses indicate matching facial features (e.g., eye shape, jawline). "
            "Be concise."
        )

        self.final_vlm_prompt_template = (
            "You are given two face images and a summary of how witnesses answered yes/no questions "
            "about biometric facial features.\n\n"
            "Witness answers summary:\n{dialogue_summary}\n\n"
            "Using this summary and the two face images ({img_token1} and {img_token2}), "
            "decide if they show the **same person** or **different people**. Briefly explain why."
        )

    @torch.no_grad()
    def play(self, img1, img2):
        chat_context = [(self.instruction_prompt, "Understood. Here's my first question:\nIs the person in the image male?")]
        history = []
        round_count = 0

        for round_idx in range(self.max_rounds):
            question = chat_context[-1][1]

            print(f"\n🕵️ Round {round_idx + 1}: {question.strip()}")

            ans1 = self.vlm_model.inference(
                qs=f"Just answer yes or no.\n{question} {self.image_token}",
                img_files=[img1],
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8
            )[0]

            ans2 = self.vlm_model.inference(
                qs=f"Just answer yes or no.\n{question} {self.image_token}",
                img_files=[img2],
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8
            )[0]

            print(f"👤 Witness #1: {ans1.strip()}")
            print(f"👤 Witness #2: {ans2.strip()}")

            history.append((question, ans1, ans2))
            if "none" in question.lower():
                break

            chat_context.append((question, f"Witness 1: {ans1.strip()}\nWitness 2: {ans2.strip()}"))

            dialogue = "\n".join(f"Q: {q}\nA1: {a1}\nA2: {a2}" for q, a1, a2 in history)

            next_question = self.llm_model.text_to_text(
                system_prompt=self.instruction_prompt,
                prompt=dialogue + "\nWhat is your next yes/no question? If you're confident, say 'None'."
            )[0]
            print(next_question)
            chat_context.append(("What is your next yes/no question? If you're confident, say 'None'.", next_question))
            round_count += 1

        history_text = "\n".join(
            f"Q: {q.strip()}\nA1: {a1.strip()}\nA2: {a2.strip()}" for q, a1, a2 in history
        )
        dialogue_summary = self.llm_model.text_to_text(
            system_prompt=self.summarize_prompt,
            prompt=history_text
        )

        final_prompt = self.final_vlm_prompt_template.format(
            dialogue_summary=dialogue_summary,
            img_token1=self.image_token,
            img_token2=self.image_token
        )

        final_decision = self.vlm_model.inference(
            qs=final_prompt,
            img_files=[img1, img2],
            num_return_sequences=1,
            do_sample=False,
            temperature=0
        )[0].replace("\n", "")

        print(f"\n🧠 VLM Final Decision: {final_decision}")
        return final_decision, history, round_count, dialogue_summary

    

        

    
    
    
    
           
        

        
         
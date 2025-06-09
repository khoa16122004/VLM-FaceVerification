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
            "You will receive multiple brief opinions on a binary question.\n"
            "Treat these as votes and determine the majority viewpoint.\n"
            "Summarize the overall consensus in a short sentence, focusing only on the main idea."
        )
        
        conclusion_summarize_prompt = (
            "Summarize the following multiple responses into a concise consensus statement:"
        )

        conclusion_prompt_template = (
            "You are given two face images and a summary of expert opinions comparing their biometric features, "
            "Based on this summary and the visual content of the two images, provide a conclusion about whether they likely "
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
            "🕵️ Your Role: You're a detective who **cannot see the images**. Each witness sees a **different image**.\n"
            "Your task is to ask both witnesses the **same yes/no question** about **biometric facial features**.\n\n"
            "🎯 GAME RULES:\n"
            "- Only ask clear yes/no questions.\n"
            "- Focus on **objective facial features** like eye shape, nose length, jaw structure, etc.\n"
            "- Do NOT ask subjective or comparative questions.\n"
            "- Avoid questions that cannot be answered from a single image (e.g., 'Is this person the same as the other?').\n"
            "- After each question, compare their answers. If you're confident, say 'None' to stop.\n\n"
            "🧠 Example questions:\n"
            "- Does the person have a sharp jawline?\n"
            "- Are the eyes almond-shaped?\n"
            "- Does the person have a wide nose?\n"
            "- Is the upper lip significantly thicker than the lower lip?\n"
        )

        self.summarize_prompt = (
            "Given the detective’s question-and-answer history, summarize how similar or different the witnesses' answers were. "
            "Highlight whether their responses suggest **biometric similarity** (e.g., same eye shape, jawline) or not. "
            "Be concise and focus on **whether the facial features described appear to match** across both images."
        )

        self.final_vlm_prompt_template = (
            "You are given two face images and a summary of how witnesses answered yes/no questions "
            "about various biometric facial features (such as eyes, nose, jawline, etc).\n\n"
            "Summary of witness answers:\n{dialogue_summary}\n\n"
            "Now, based on this summary and the visual appearance of the two images ({img_token1} and {img_token2}), "
            "conclude whether the two images likely show the **same individual** or **different individuals**. Justify your conclusion briefly."
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

    

        

    
    
    
    
           
        

        
         
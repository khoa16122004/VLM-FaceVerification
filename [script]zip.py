import zipfile
import os

def zip_folder(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

# Ví dụ sử dụng
zip_folder(r'controller=traditional_vlm=llava-next-interleave-qwen-7b_llm=Llama-7b_num_samples=9', 
           'llava_next_llama_7b_9.zip')
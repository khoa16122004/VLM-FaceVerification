import zipfile
import os

def unzip_lfw(zip_path="lfw_preprocess.zip", extract_to="lfw_preprocess"):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"✅ Extracted '{zip_path}' to '{extract_to}'")

if __name__ == "__main__":
    unzip_lfw()

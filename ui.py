import streamlit as st
from face_verification import FaceVerification
from utils import CustomDataset
from get_arch import init_lvlm_model
from PIL import Image

# Load dataset và model
dataset = CustomDataset(root_dir="samples", type="different")

pretrained_lvlm = "llava-next-interleave-qwen-7b"
model_name_lvlm = "llava_qwen"
vlm_model = (pretrained_lvlm, model_name_lvlm)
llm_model = None

traditional_controller = FaceVerification(vlm_model=vlm_model, llm_model=llm_model)

st.set_page_config(page_title="Face Verification Demo", layout="centered")

st.title("🔍 Face Verification Inference")
sample_id = st.number_input("Enter sample ID:", min_value=0, max_value=len(dataset)-1, step=1)

# Load sample
label, case_name, (img1, img2), (img1_path, img2_path) = dataset[sample_id]

st.markdown(f"**Case Name:** {case_name}")
st.markdown(f"**Label:** {label}")
st.markdown(f"**Image 1 Path:** `{img1_path}`")
st.markdown(f"**Image 2 Path:** `{img2_path}`")

col1, col2 = st.columns(2)
with col1:
    st.image(Image.open(img1_path), caption="Image 1", use_column_width=True)
with col2:
    st.image(Image.open(img2_path), caption="Image 2", use_column_width=True)

st.markdown("---")

# Chọn chế độ inference
mode = st.radio("Select Inference Mode:", ["Direct Answer", "Explain Answer"])
if st.button("Run Inference"):
    direct = 1 if mode == "Direct Answer" else 0
    response = traditional_controller.simple_answer(img1, img2, direct_return=direct)
    st.success("✅ Inference Result:")
    st.write(response)

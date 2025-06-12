import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import streamlit as st
from face_verification import FaceVerification
from utils import CustomDataset
from get_arch import init_lvlm_model
from PIL import Image
import torch

@st.cache_resource
def load_controller():
    dataset = CustomDataset(root_dir="samples", type="same")
    pretrained_lvlm = "llava-onevision-qwen2-7b-ov"
    model_name_lvlm = "llava_qwen"
    vlm_model = (pretrained_lvlm, model_name_lvlm)
    llm_model = ("Llama-7b", )
    controller = FaceVerification(vlm_model=vlm_model, llm_model=llm_model)
    return controller, dataset

st.set_page_config(page_title="Face Verification Demo", layout="wide")  # wide layout để rộng ngang hơn
st.title("🔍 Face Verification Inference")

traditional_controller, dataset = load_controller()

sample_id = st.number_input("Enter sample ID:", min_value=0, max_value=len(dataset)-1, step=1)
label, case_name, (img1, img2), (img1_path, img2_path) = dataset[sample_id]

st.markdown(f"**Case Name:** {case_name}")
st.markdown(f"**Label:** {label}")
st.markdown(f"**Image 1 Path:** `{img1_path}`")
st.markdown(f"**Image 2 Path:** `{img2_path}`")

col1, col2, col3 = st.columns([1,1,1])  # 3 cột bằng nhau

with col1:
    st.image(Image.open(img1_path), caption="Image 1", use_column_width=True)
with col2:
    st.image(Image.open(img2_path), caption="Image 2", use_column_width=True)

with col3:
    mode = st.radio("Select Inference Mode:", ["Run Both Modes", "Run Sampling Mode", "Run All"])
    if st.button("Run Inference"):
        if mode == "Run Both Modes" or mode == "Run All":
            with st.spinner("Running Direct Answer..."):
                direct_response = traditional_controller.simple_answer(img1, img2, direct_return=1)
            with st.spinner("Running Explain Answer..."):
                explain_response = traditional_controller.simple_answer(img1, img2, direct_return=0)

            st.success("✅ Direct & Explain Results:")
            st.subheader("🟢 Direct Answer")
            st.write(direct_response)

            st.subheader("🔵 Explain Answer")
            st.write(explain_response)

        if mode == "Run Sampling Mode" or mode == "Run All":
            sampling_answer = st.text_input("Enter your Sampling Answer (Yes/No):", value="Yes")
            if sampling_answer:
                with st.spinner("Running Sampling Answer..."):
                    final_decision, all_question_responses, selection_responses, summarized_responses = traditional_controller.sampling_answer(img1, img2, sampling_answer)

                st.success("✅ Sampling Result:")

                st.subheader("🧠 Final Decision")
                st.write(final_decision)

                st.subheader("📋 All Question Responses")
                for idx, response in enumerate(all_question_responses):
                    st.markdown(f"**Q{idx+1}:** {response}")

                st.subheader("✅ Selected Responses")
                for idx, sel in enumerate(selection_responses):
                    st.markdown(f"**Selection {idx+1}:** {sel}")

                st.subheader("📝 Summarized Responses")
                for idx, summary in enumerate(summarized_responses):
                    st.markdown(f"**Summary {idx+1}:** {summary}")

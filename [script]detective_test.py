import streamlit as st
from PIL import Image
from face_verification import DetectiveGame  # lớp đã thiết kế
from utils import CustomDataset  # class dataset của bạn
import random

# 🖼️ Load dataset
dataset = CustomDataset(root_dir="samples", type="different")
sample_ids = list(range(len(dataset)))

# 💾 Cache để không load lại ảnh liên tục
@st.cache_data
def load_sample(sample_id):
    label, case_name, (img1, img2), (img1_path, img2_path) = dataset[sample_id]
    return (img1_path, img2_path), (img1, img2), label

# 🕵️‍♀️ Tạo detective game (cần truyền model của bạn vào đây)
@st.cache_resource
def get_game():
    pretrained_lvlm = "llava-next-interleave-qwen-7b"
    model_name_lvlm = "llava_qwen"
    vlm_model = (pretrained_lvlm, model_name_lvlm)

    llm_model = ("Llama-7b", )

    # ciontroller
    controller = DetectiveGame(vlm_model=vlm_model, 
                               llm_model=llm_model)
    
    return controller

st.set_page_config(page_title="🕵️ Detective Game", layout="wide")
st.title("🕵️ Detective Game: Are They the Same Person?")

# 🔢 Chọn sample
sample_id = st.selectbox("Select Sample ID:", sample_ids)
(img1_path, img2_path), (img1, img2), ground_truth = load_sample(sample_id)

# 🖼️ Hiển thị ảnh
col1, col2 = st.columns(2)
with col1:
    st.image(img1, caption="Witness #1 Image", use_column_width=True)
with col2:
    st.image(img2, caption="Witness #2 Image", use_column_width=True)

# 🎮 Chế độ chơi
mode = st.radio("Select Mode:", ["🔁 LLM Asks Questions", "🙋 You Ask Questions"])

# 🎲 Khởi tạo game
game = get_game()

if "history" not in st.session_state:
    st.session_state.history = []
if "finished" not in st.session_state:
    st.session_state.finished = False
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "final_decision" not in st.session_state:
    st.session_state.final_decision = ""

# 🔁 LLM chơi tự động
if mode == "🔁 LLM Asks Questions":
    if st.button("Start Game"):
        with st.spinner("Detective is asking questions..."):
            decision, history, round_count, summary = game.play(img1, img2)
            st.session_state.history = history
            st.session_state.summary = summary
            st.session_state.final_decision = decision
            st.session_state.finished = True

# 🙋 Người dùng hỏi thủ công
if mode == "🙋 You Ask Questions":
    user_question = st.text_input("Enter your yes/no question:")
    if st.button("Ask Question") and user_question.strip():
        # Gọi VLM trả lời cho mỗi ảnh
        with st.spinner("Witnesses answering..."):
            ans1 = game.vlm.inference(
                qs=f"Just answer yes or no.\n{user_question} {game.image_token}",
                img_files=[img1],
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8
            )[0]
            ans2 = game.vlm.inference(
                qs=f"Just answer yes or no.\n{user_question} {game.image_token}",
                img_files=[img2],
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8
            )[0]

            st.session_state.history.append((user_question, ans1.strip(), ans2.strip()))

    if st.button("Finish and Summarize"):
        with st.spinner("Summarizing and deciding..."):
            history_text = "\n".join(
                f"Q: {q}\nA1: {a1}\nA2: {a2}" for q, a1, a2 in st.session_state.history
            )
            summary = game.llm.text_to_text(
                system_prompt=game.summarize_prompt,
                prompt=history_text
            )
            final_prompt = game.final_vlm_prompt_template.format(
                dialogue_summary=summary,
                img_token1=game.image_token,
                img_token2=game.image_token
            )
            final_decision = game.vlm.inference(
                qs=final_prompt,
                img_files=[img1, img2],
                num_return_sequences=1,
                do_sample=False,
                temperature=0
            )[0].replace("\n", "")
            st.session_state.summary = summary
            st.session_state.final_decision = final_decision
            st.session_state.finished = True

# 🧾 Hiển thị kết quả
if st.session_state.history:
    st.subheader("📜 Question & Answer History")
    for i, (q, a1, a2) in enumerate(st.session_state.history):
        st.markdown(f"**Round {i+1}**")
        st.markdown(f"🕵️ Q: {q}")
        st.markdown(f"👤 Witness #1: {a1}")
        st.markdown(f"👤 Witness #2: {a2}")

if st.session_state.finished:
    st.subheader("📝 Summary from LLM")
    st.markdown(st.session_state.summary)

    st.subheader("🧠 VLM Final Decision")
    st.success(st.session_state.final_decision)

    st.info(f"✅ Ground Truth: {'Same' if ground_truth == 1 else 'Different'}")
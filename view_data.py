import streamlit as st
from PIL import Image
from your_dataset_file import LFW  # thay thế bằng đúng đường dẫn file chứa class LFW

@st.cache_resource
def load_dataset(img_dir, pair_path):
    return LFW(
        IMG_DIR=img_dir,
        PAIR_PATH=pair_path,
        transform=None
    )

# UI
st.title("LFW Sample Viewer")

img_dir = st.text_input("Path to image directory", "lfw/images")
pair_path = st.text_input("Path to pairs.txt", "lfw/pairs.txt")

dataset = load_dataset(img_dir, pair_path)

sample_id = st.number_input("Enter sample ID", min_value=0, max_value=len(dataset) - 1, step=1)

if st.button("Show Sample"):
    img1, img2, label = dataset[sample_id]

    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Image 1", use_column_width=True)
    with col2:
        st.image(img2, caption="Image 2", use_column_width=True)

    st.markdown(f"### Label: `{label}`")

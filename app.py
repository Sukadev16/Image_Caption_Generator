import streamlit as st
import numpy as np
import pickle

from PIL import Image

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Caption Generator")
st.write("Upload an image and generate a caption using CNN + LSTM")


# ==============================
# LOAD TRAINED MODEL
# ==============================

caption_model = load_model("caption_model_trained.keras")


# ==============================
# LOAD TOKENIZER
# ==============================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


max_length = 34


# ==============================
# LOAD RESNET50 FEATURE EXTRACTOR
# ==============================

base_model = ResNet50(weights='imagenet')

feature_model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)


# ==============================
# CONVERT INDEX TO WORD
# ==============================

def idx_to_word(integer):

    for word, index in tokenizer.word_index.items():

        if index == integer:

            return word

    return None


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = feature_model.predict(
        image,
        verbose=0
    )

    return feature


# ==============================
# PREDICT CAPTION
# ==============================

def predict_caption(feature):

    in_text = "startseq"

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length
        )

        yhat = caption_model.predict(
            [feature, sequence],
            verbose=0
        )

        yhat = np.argmax(yhat)

        word = idx_to_word(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    caption = in_text.replace("startseq", "")
    caption = caption.replace("endseq", "")
    caption = caption.strip()

    return caption


# ==============================
# FILE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)


# ==============================
# DISPLAY IMAGE AND GENERATE
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating Caption..."):

            feature = extract_features(image)

            caption = predict_caption(feature)

        st.success("Caption Generated Successfully")

        st.markdown("### Generated Caption")

        st.write(caption)
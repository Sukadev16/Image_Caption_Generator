import pickle
import pandas as pd
import numpy as np

from nltk.translate.meteor_score import meteor_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("caption_model_trained.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load features
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# Load captions
captions_df = pd.read_csv("data/captions.txt")

max_length = 34


def idx_to_word(integer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image_feature):

    in_text = "startseq"

    for _ in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length
        )

        yhat = model.predict(
            [image_feature, sequence],
            verbose=0
        )

        yhat = np.argmax(yhat)

        word = idx_to_word(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text


scores = []

for image_id in list(features.keys())[:500]:

    refs = captions_df[
        captions_df["image"] == image_id
    ]["caption"].tolist()

    pred = predict_caption(
        model,
        features[image_id]
    )

    pred_tokens = pred.split()

    ref_tokens = [r.split() for r in refs]

    score = meteor_score(
        ref_tokens,
        pred_tokens
    )

    scores.append(score)

print("\nMETEOR Score:")
print(np.mean(scores))
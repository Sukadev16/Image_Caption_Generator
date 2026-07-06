import pickle
import pandas as pd
import numpy as np

from nltk.translate.bleu_score import corpus_bleu
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

    for i in range(max_length):

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

    return in_text.split()


actual = []
predicted = []


for image_id in list(features.keys())[:500]:

    temp = captions_df[
        captions_df["image"] == image_id
    ]["caption"].tolist()

    references = [caption.split() for caption in temp]

    yhat = predict_caption(
        model,
        features[image_id]
    )

    actual.append(references)

    predicted.append(yhat)


print("\nBLEU Scores")

print("BLEU-1:",
      corpus_bleu(actual,
                  predicted,
                  weights=(1.0,0,0,0)))

print("BLEU-2:",
      corpus_bleu(actual,
                  predicted,
                  weights=(0.5,0.5,0,0)))

print("BLEU-3:",
      corpus_bleu(actual,
                  predicted,
                  weights=(0.3,0.3,0.3,0)))

print("BLEU-4:",
      corpus_bleu(actual,
                  predicted,
                  weights=(0.25,0.25,0.25,0.25)))
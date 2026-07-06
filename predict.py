import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


print("Loading Model...")

model = load_model(
    "caption_model_trained.keras"
)

print("Model Loaded Successfully")


with open("tokenizer.pkl", "rb") as f:

    tokenizer = pickle.load(f)

print("Tokenizer Loaded Successfully")


with open("features.pkl", "rb") as f:

    features = pickle.load(f)

print("Features Loaded Successfully")


max_length = 34


def idx_to_word(integer):

    for word, index in tokenizer.word_index.items():

        if index == integer:

            return word

    return None



def predict_caption(model, image_feature):

    in_text = "startseq"

    used_words = set()

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences(
            [in_text]
        )[0]

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

        if word == "endseq":
            break

        if word in used_words:
            break

        used_words.add(word)

        in_text += " " + word

    return in_text



image_name = "1000268201_693b08cb0e.jpg"

print("\nImage :", image_name)

feature = features[image_name]

print("\nGenerating Caption...")


caption = predict_caption(

    model,

    feature

)

caption = caption.replace(

    "startseq",

    ""

).strip()


print("\nPredicted Caption :")

print(caption)
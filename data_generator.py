import pandas as pd
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


with open("features.pkl", "rb") as f:
    features = pickle.load(f)


vocab_size = len(tokenizer.word_index) + 1

max_length = 34


def load_captions():

    df = pd.read_csv(
        "data/captions.txt"
    )

    return df


captions_df = load_captions()


def data_generator(batch_size=64):

    X1, X2, y = [], [], []

    count = 0

    while True:

        for _, row in captions_df.iterrows():

            image_id = row["image"]

            caption = row["caption"]

            seq = tokenizer.texts_to_sequences(
                [caption]
            )[0]

            for i in range(1, len(seq)):

                in_seq = seq[:i]

                out_seq = seq[i]

                in_seq = pad_sequences(
                    [in_seq],
                    maxlen=max_length
                )[0]

                out_seq = to_categorical(
                    [out_seq],
                    num_classes=vocab_size
                )[0]

                X1.append(features[image_id][0])

                X2.append(in_seq)

                y.append(out_seq)

                count += 1

                if count == batch_size:

                    yield (

                        (

                            np.array(X1),

                            np.array(X2)

                        ),

                        np.array(y)

                    )

                    X1, X2, y = [], [], []

                    count = 0
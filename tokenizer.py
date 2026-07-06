import pandas as pd
import re
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer


def clean_caption(caption):

    caption = caption.lower()

    caption = re.sub(r'[^a-z ]', '', caption)

    caption = caption.strip()

    caption = "startseq " + caption + " endseq"

    return caption


def load_captions(file_path):

    df = pd.read_csv(
        file_path,
        names=['image', 'caption']
    )

    if df.iloc[0]['image'] == 'image':

        df = df.iloc[1:]

    df['caption'] = df['caption'].apply(clean_caption)

    return df


if __name__ == "__main__":

    captions_file = "data/captions.txt"

    df = load_captions(captions_file)

    captions = df['caption'].tolist()

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(captions)

    vocab_size = len(tokenizer.word_index) + 1

    max_length = max(

        len(caption.split())

        for caption in captions

    )

    print("\nVocabulary Size :", vocab_size)

    print("Maximum Caption Length :", max_length)

    with open("tokenizer.pkl", "wb") as f:

        pickle.dump(tokenizer, f)

    print("\nTokenizer Saved Successfully")
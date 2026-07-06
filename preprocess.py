import pandas as pd
import re


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

    print("Original Shape :", df.shape)

    # Remove header row if present

    if df.iloc[0]['image'] == 'image':

        df = df.iloc[1:]

    df['caption'] = df['caption'].apply(clean_caption)

    print("\nSample Processed Captions:\n")

    for i in range(5):

        print(df.iloc[i]['image'])

        print(df.iloc[i]['caption'])

        print()

    return df


if __name__ == "__main__":

    captions_file = "data/captions.txt"

    captions_df = load_captions(captions_file)

    print("\nTotal Images :",
          captions_df['image'].nunique())

    print("Total Captions :",
          len(captions_df))
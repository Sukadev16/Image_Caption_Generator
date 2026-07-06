import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer

from model import define_model


# Load tokenizer

with open("tokenizer.pkl", "rb") as f:

    tokenizer = pickle.load(f)


# Load features

with open("features.pkl", "rb") as f:

    features = pickle.load(f)


# Vocabulary size

vocab_size = len(tokenizer.word_index) + 1

max_length = 37


print("Vocabulary Size :", vocab_size)

print("Maximum Caption Length :", max_length)

print("Total Image Features :", len(features))


# Create model

model = define_model(

    vocab_size,

    max_length

)


print("\nModel Created Successfully")


# IMPORTANT

# Since your laptop has 8GB RAM,

# we are NOT training for many epochs now.


model.save(

    "caption_model.keras"

)


print("\nModel Saved Successfully")
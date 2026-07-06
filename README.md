# Image Caption Generator using CNN + LSTM

## Overview

This project generates captions for images using:

- ResNet50 CNN
- LSTM
- Flickr8K Dataset
- Streamlit

---

## Dataset

Flickr8K Dataset

- Images : 8091
- Captions : 40455
- Vocabulary Size : 8768

---

## Model Architecture

CNN (ResNet50)
↓

2048 Feature Vector
↓

Dense(256)

Caption Sequence
↓

Embedding
↓

LSTM(256)

↓

Add Layer

↓

Dense

↓

Softmax

↓

Generated Caption

---

## Results

BLEU-1 : 0.4986

BLEU-2 : 0.3370

BLEU-3 : 0.2540

BLEU-4 : 0.1345

METEOR : 0.3353

---

## Streamlit App

```bash
streamlit run app.py
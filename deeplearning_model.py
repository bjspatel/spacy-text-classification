import spacy
from spacy.util import minibatch, compounding, decaying
import random
import torch

import pandas as pd


def set_gpu():
    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")


def load_data():
    # Load CSV file
    df = pd.read_csv("data/flair_labeled_sentiments.csv", names=['feedback', 'text'])

    df['feedback'] = df['feedback'].replace('__label__Negative', 0)
    df['feedback'] = df['feedback'].replace('__label__Positive', 1)

    # The features we want to analyze
    df_texts = df['text'].values

    # The labels, or answers, we want to test against
    df_labels = df['feedback'].values

    df_train_data = [
        (df_texts[index], {"cats": {"POSITIVE": y, "NEGATIVE": abs(1-y)}})
        for index, y in enumerate(df_labels)
    ]

    random.shuffle(df_train_data)

    split = int(len(df_train_data) * 0.8)

    return df_train_data[:split], df_train_data[split:]


def get_batches(train_data, model_type):
    max_batch_sizes = {
        "tagger": 32,
        "parser": 16,
        "ner": 16,
        "textcat": 64
    }
    max_batch_size = max_batch_sizes[model_type]
    if len(train_data) < 1000:
        max_batch_size /= 2
    if len(train_data) < 500:
        max_batch_size /= 2
    batch_size = compounding(1, max_batch_size, 1.001)
    batches = minibatch(train_data, size=batch_size)
    return batches


def create_model():
    set_gpu()

    TRAIN_DATA, TEST_DATA = load_data()

    nlp = spacy.load("en_pytt_bertbaseuncased_lg")

    textcat = nlp.create_pipe("pytt_textcat", config={"exclusive_classes": True})

    for label in ("POSITIVE", "NEGATIVE"):
        textcat.add_label(label)

    nlp.add_pipe(textcat)

    optimizer = nlp.resume_training()

    dropout = decaying(0.6, 0.2, 1e-4)

    print("Training the model...")

    for i in range(10):
        print("Iteration =>", i)
        random.shuffle(TRAIN_DATA)
        losses = {}
        for batch in get_batches(TRAIN_DATA, "textcat"):
            texts, cats = zip(*batch)
            print(texts, cats)
            nlp.update(texts, cats, sgd=optimizer, losses=losses, drop=dropout)
        print(i, losses)

    with nlp.use_params(optimizer.averages):
        nlp.to_disk("models")


if __name__ == "__main__":
    # create_model()
    test_text = "happy"
    print("Loading from", "models")
    nlp2 = spacy.load("models")
    doc2 = nlp2(test_text)
    print(test_text, doc2.cats)

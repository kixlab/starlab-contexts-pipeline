from sentence_transformers import SentenceTransformer
from numpy import zeros

model = SentenceTransformer("all-MiniLM-L6-v2")

def bert_embedding(texts):
    if len(texts) == 0:
        return []

    for i in range(len(texts)):
        if texts[i] == "":
            texts[i] = " "
    embeddings = model.encode(texts)
    return embeddings
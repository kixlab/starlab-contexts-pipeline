import nltk

from sklearn.decomposition import PCA

def standardize(embeddings):
    std = embeddings.std(axis=0)
    mean = embeddings.mean(axis=0)
    if (std == 0).any():
        return embeddings
    return (embeddings - mean) / std

def reduce_dim(fit_embeddings, transform_embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(fit_embeddings)
    reduced_embeddings = pca.transform(transform_embeddings)

    return reduced_embeddings, pca

from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import zeros
from nltk.corpus import stopwords

nltk.download('stopwords')

def tfidf_embedding(texts): 
    if len(texts) == 0:
        return []
    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        ngram_range=(1, 2), max_features=1000
    )
    for i in range(len(texts)):
        if texts[i] == "":
            texts[i] = " "

    X = vectorizer.fit_transform(texts)
    embeddings = X.toarray()
    return embeddings

        
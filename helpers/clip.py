import torch
import clip
from PIL import Image

def clip_embed_image(image_paths):
    """
    Embed an image using a CLIP model
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    ### transform the image to a tensor
    images = torch.cat([preprocess(Image.open(path)).unsqueeze(0) for path in image_paths]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def clip_embed_text(texts):
    """
    Embed a text using a CLIP model
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # prefix = "A photo of "
    prefix = ""

    embeddings = torch.cat([clip.tokenize([prefix + text.lower()]) for text in texts]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(embeddings)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def clip_similar_per_text(texts, image_paths, top_k=1):
    ### TODO: May need to consider the narrations for each image
    """
    Find the most similar image to a given text
    if multiple texts are given, return the most similar image to each text
    if multiple images are equally similar to a text, return the last one
    """
    image_embeddings = clip_embed_image(image_paths)
    text_embeddings = clip_embed_text(texts)

    similarity = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=0)
    values, indices = similarity.T.topk(top_k)

    indices = indices.squeeze(1).tolist()
    if top_k == 1:
        return [image_paths[idx] for idx in indices]
    else:
        return [[image_paths[idx] for idx in idxs] for idxs in indices]
    

class ClipModel:
    image_paths = []
    image_embeddings = []

    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.image_embeddings = clip_embed_image(image_paths)

    def find_similar_per_text(self, texts, top_k=1):
        text_embedding = clip_embed_text(texts)
        similarity = (self.image_embeddings @ text_embedding.T).softmax(dim=0)
        values, indices = similarity.T.topk(1)
        # print(values, indices)
        indices = indices.squeeze(1).tolist()
        if top_k == 1:
            return [self.image_paths[idx] for idx in indices]
        else:
            return [[self.image_paths[idx] for idx in idxs] for idxs in indices]
        
    def get_scores_per_text(self, texts):
        text_embedding = clip_embed_text(texts)
        similarity = (self.image_embeddings @ text_embedding.T).softmax(dim=0)
        return similarity.tolist()
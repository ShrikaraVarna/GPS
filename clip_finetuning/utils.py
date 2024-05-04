import torch.nn as nn
#Reference from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def find_most_similar_embeddings(embeddings1, embeddings2):
    # Normalize the embeddings to unit vectors
    embeddings1_normalized = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_normalized = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Compute the cosine similarities
    cosine_similarities = np.dot(embeddings1_normalized, embeddings2_normalized.T)

    # Find the index of the maximum cosine similarity in each row
    most_similar_indices = np.argmax(cosine_similarities, axis=1)

    return most_similar_indices

def inference_image_match_score(dataloader):
    pbar = tqdm(dataloader, total=len(dataloader))
    image_embeddings = []
    text_embeddings = []
    targets = torch.arange(len(dataloader), device=device)
    with torch.no_grad():
        for batch in pbar:
            inputs = processor(text=texts, images=images, return_tensors='pt', padding=True)
            outputs = model(**inputs)
            print(outputs)
            image_embeddings, text_embeddings = outputs['image_embeds'], outputs['text_embeds']
            image_embeddings.extend(image_embeddings)
            text_embeddings.extend(text_embeddings)
    image_embeddings, text_embeddings = torch.concat(image_embeddings), torch.concat(text_embeddings)

    predicted = find_most_similar_embeddings(image_embeddings, text_embeddings)
    image_match_score = (targets == predicted).mean()
    return image_match_score

def clean_images(dataset):
    image_paths, list_txts = dataset['image'], [conv[0]['value'] for conv in dataset['conversations']]
    valid_image_paths = []
    valid_list_txts = []

    # Iterate over the image paths and corresponding texts
    for image_path, text in zip(image_paths, list_txts):
        if os.path.exists(image_path):
            valid_image_paths.append(image_path)
            valid_list_txts.append(text)
        else:
            print(f"Warning: The file {image_path} does not exist and will be removed from the dataset.")

    return valid_image_paths, valid_list_txts

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, image_features, text_features):
        # Normalize features to get unit vectors
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        text_features = nn.functional.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity
        logits = torch.matmul(image_features, text_features.t()) / self.temperature

        # Targets: diagonal elements are the positive samples
        targets = torch.arange(logits.shape[0], device=logits.device)
        return nn.functional.cross_entropy(logits, targets)


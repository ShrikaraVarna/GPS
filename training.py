import json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from datasets import load_dataset

device='cuda' if torch.cuda.is_available else 'cpu'
##########DATALOADERS#############
dataset = load_dataset("Luckyjhg/Geo170K", split="alignment") #size is 60k
train_test_split = dataset.train_test_split(test_size=0.2)
train_val_split = train_test_split['train'].train_test_split(test_size=0.2)

train_dataset, val_dataset, test_dataset = train_val_split['train'], train_val_split['test'], train_test_split['test'] 
train_image_paths, train_list_txt = train_dataset['image'],  [conv[0]['value'] for conv in train_dataset['conversations']]
val_image_paths, val_list_txt = val_dataset['image'],  [conv[0]['value'] for conv in val_dataset['conversations']]
test_image_paths, test_list_txt = test_dataset['image'],  [conv[0]['value'] for conv in test_dataset['conversations']]

class GeometryImageCaptionDataset():
    def __init__(self, list_image_path,list_txt):
        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

CLIPTrainDataset = GeometryImageCaptionDataset(train_image_paths, train_list_txt)
CLIPValDataset = GeometryImageCaptionDataset(val_image_paths, val_list_txt)
CLIPTestDataset = GeometryImageCaptionDataset(test_image_paths, test_list_txt)

train_dataloader = DataLoader(CLIPTrainDataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(CLIPValDataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(CLIPTestDataset, batch_size=64, shuffle=True)


#################### MODEL CONFIG####################
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, image_features, text_features):
        # Normalize features to get unit vectors
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity
        logits = torch.matmul(image_features, text_features.T) / self.temperature

        # Targets: diagonal elements are the positive samples
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
loss_fn = ContrastiveLoss()

############## UTILS #######################
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
            image_embeddings = model(images)
            text_embeddings = model(texts)
            image_embeddings.extend(image_embeddings)
            text_embeddings.extend(text_embeddings)
    image_embeddings, text_embeddings = torch.concat(image_embeddings), torch.concat(text_embeddings)

    predicted = find_most_similar_embeddings(image_embeddings, text_embeddings)
    image_match_score = (targets == predicted).mean()
    return image_match_score
    
def possible_forward_loss(text_embeddings, image_embeddings):
    #Target calculation for retrieval - This is XEntLoss
    logits = (text_embeddings @ image_embeddings.T ) / temperature
    image_similarity = image_embeddings @ image_embeddings.T
    text_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax((image_similarity + text_similarity)/2*temperature, dim=-1)

    #Optionally, targets can be 
    #targets = torch.arange(len(images), device=device)

    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    total_loss = loss.mean()

##################### TRAINING #####################
test_image_match_score = inference_image_match_score(test_dataloader)
print("Before training test image match score:", test_image_match_score)
num_epochs = 30
model.to(device)
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch 
        
        images= images.to(device)
        texts = texts.to(device)

        image_embeddings, text_embeddings = model(images, texts)
        total_loss = loss_fn(image_embeddings, text_embeddings)

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

    image_match_score = inference_image_match_score(val_dataloader)
    print("Validation image match score: ", image_match_score)
    
            
####################INFERENCE#################
test_image_match_score = inference_image_match_score(test_dataloader)
print("Final test image match score:", test_image_match_score)

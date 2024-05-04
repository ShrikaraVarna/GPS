import json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import clip
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from datasets import load_dataset
from utils import *

device='cuda' if torch.cuda.is_available else 'cpu'
##########DATALOADERS#############
dataset = load_dataset("Luckyjhg/Geo170K", split="alignment") #size is 60k
train_test_split = dataset.train_test_split(test_size=0.2)
train_val_split = train_test_split['train'].train_test_split(test_size=0.2)

train_dataset, val_dataset, test_dataset = train_val_split['train'], train_val_split['test'], train_test_split['test'] 
train_image_paths, train_list_txts = clean_images(train_dataset)
val_image_paths, val_list_txts = clean_images(val_dataset)
test_image_paths, test_list_txts = clean_images(test_dataset)

class GeometryImageCaptionDataset():
    def __init__(self, list_image_path,list_txt):
        self.image_path = list_image_path
        self.title  = list_txt

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        title = self.title[idx]
        return image, title

CLIPTrainDataset = GeometryImageCaptionDataset(train_image_paths, train_list_txt)
CLIPValDataset = GeometryImageCaptionDataset(val_image_paths, val_list_txt)
CLIPTestDataset = GeometryImageCaptionDataset(test_image_paths, test_list_txt)

train_dataloader = DataLoader(CLIPTrainDataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(CLIPValDataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(CLIPTestDataset, batch_size=64, shuffle=True)


#################### MODEL CONFIG####################

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
loss_fn = ContrastiveLoss()


##################### TRAINING #####################
test_image_match_score = inference_image_match_score(test_dataloader)
print("Before training test image match score:", test_image_match_score)
exit()
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

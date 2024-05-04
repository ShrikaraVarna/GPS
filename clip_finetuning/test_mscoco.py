import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from pycocotools.coco import COCO
from torch.nn import functional as F
import os
import numpy as np
from utils import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize COCO API for instance annotations
dataDir = '/data/tir/projects/tir7/user_data/sterupal/coco'
dataType = 'val2017'
annFile = f'{dataDir}/annotations/captions_{dataType}.json'

# Setup the MSCOCO dataset
class CocoCaptions(datasets.CocoCaptions):
    def __init__(self, root, annFile, transform=None):
        super(CocoCaptions, self).__init__(root, annFile, transform=transform)

    def __getitem__(self, index):
        img, captions = super(CocoCaptions, self).__getitem__(index)
        return img, captions[0]  # Use the first caption

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit CLIP input dimensions
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to match CLIP's expected input
])

# Load the full dataset
full_coco_dataset = CocoCaptions(root=f'{dataDir}/{dataType}', annFile=annFile, transform=transform)
# Select 10% of the dataset randomly
#subset_indices = np.random.choice(len(full_coco_dataset), size=int(0.1 * len(full_coco_dataset)), replace=False)
#coco_dataset = Subset(full_coco_dataset, subset_indices)
data_loader = DataLoader(full_coco_dataset, batch_size=64, shuffle=True)

# Load CLIP model
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model.to(device)

model, processor = clip_model, clip_processor
accuracy = inference_image_match_score(dataloader)
exit()
# Training configuration
optimizer = optim.Adam(clip_model.parameters(), lr=5e-6)
temperature = 0.07

# Contrastive Loss Function
def contrastive_loss(image_features, text_features, temperature):
    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)

# Training loop
def train(model, data_loader, processor, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, captions in data_loader:
            images = images.to(device)
            inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)

            # Forward pass
            image_features = model.get_image_features(images)
            text_features = model.get_text_features(**inputs)

            loss = criterion(image_features, text_features, temperature)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

num_epochs = 5
train(clip_model, data_loader, clip_processor, contrastive_loss, optimizer, num_epochs)


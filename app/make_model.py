from embedders import Embedder_CLIP_ViT, Embedder_Poses, Embedder_Raw, Embedder_VGG19
from sklearn.decomposition import IncrementalPCA
import csv
from train import train
import os


# Choose embedders and reducers, see train.py
"""
embedders = {
    "vgg19": Embedder_VGG19(reducer=IncrementalPCA(n_components=512)),
    "raw": Embedder_Raw(reducer=IncrementalPCA(n_components=512)),
    "clip_vit": Embedder_CLIP_ViT(),
    "poses": Embedder_Poses()
}
"""
embedders = {
    "vgg19": Embedder_VGG19(reducer=IncrementalPCA(n_components=512)),
    "clip_vit": Embedder_CLIP_ViT(),
}
data_root = "/data/dev.imgs.ai/ImageNet" # CSV file or folder
model_folder = "../models/ImageNet" # Where to save the model
max_data = 10000 # Limit to max_data images (useful for testing purposes)

X = []

if data_root.endswith(".csv"): # CSV
    with open(data_root, "r") as f:
        meta = csv.reader(f)
        for row in meta:
            X.append(row)
    data_root = None
else: # Not CSV
    for root, _, files in os.walk(data_root):
        for fname in files:
            X.append([os.path.relpath(os.path.join(root, fname), start=data_root), "", None])
    
if max_data: X = X[:max_data]

train(
    X=X,
    data_root=data_root, 
    model_folder=model_folder,
    embedders=embedders
)
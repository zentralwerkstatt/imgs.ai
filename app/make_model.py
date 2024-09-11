from embedders import Embedder_CLIP_ViT, Embedder_Poses, Embedder_Raw, Embedder_VGG19 # Dynamic imports
from sklearn.decomposition import IncrementalPCA # Dynamic imports
import csv
from train import collect_embed, build
import os
from util import get_current_dir, get_parent_dir, new_dir


# Choose embedders and reducers, see train.py
embedders_string = """
embedders = {
    'vgg19': Embedder_VGG19(reducer=IncrementalPCA(n_components=512)),
    'raw': Embedder_Raw(reducer=IncrementalPCA(n_components=512)),
    'clip_vit': Embedder_CLIP_ViT(),
    'poses': Embedder_Poses()
} 
"""
data_root = "/Users/fabian/Desktop/dev/imgs.ai/harvard" # Absolute, CSV file or folder, if folder need to copy to model_folder/data after training
model_folder = "static/models/Harvard"
max_data = None # Limit to max_data images (useful for testing purposes)
private = False # Whether the model is private

# Empty dataset
X = []

# Read data from folder or CSV
if data_root.endswith(".csv"): # CSV
    with open(data_root, "r") as f:
        meta = csv.reader(f)
        for row in meta:
            X.append(row)
    data_root = None
else: # Folder
    for root, _, files in os.walk(data_root):
        for fname in files:
            X.append([os.path.relpath(os.path.join(root, fname), start=data_root), "", None])

# Apply max_data
if max_data: X = X[:max_data]

# Create model folder
new_dir(model_folder)

# Write embedders to file
with open(f"{model_folder}/embedders.pytxt", "w") as f:
    f.write(embedders_string)

# Train
collect_embed(X, data_root, model_folder)
build(X, model_folder, private=private)
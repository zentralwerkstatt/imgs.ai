from train import collect_embed, build


"""
To train:
- Create a new model folder under static/models
- Create a new embedders.pytxt that contains a version of the following string
embedders = {
    'vgg19': Embedder_VGG19(reducer=IncrementalPCA(n_components=512)),
    'raw': Embedder_Raw(reducer=IncrementalPCA(n_components=512)),
    'clip_vit': Embedder_CLIP_ViT(),
    'poses': Embedder_Poses()
} 
- Move a valid metadata.csv to the new model folder, structure:
relative path to image or image url (mandatory), source url (optional), metadata column (optional), ..., metadata column (optional)
- Run this script
"""

model_folder = "static/models/Harvard" # Must have /data folder if local
max_data = None # Limit to max_data images (useful for testing purposes)
private = False # Whether the model is private
# For additional settings see train.py

collect_embed(model_folder, max_data=max_data)
build(model_folder, private=private)
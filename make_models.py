from embedders import Embedder_Poses, Embedder_VGG19, Embedder_Raw, Embedder_Face, Embedder_CLIP
from train import make_model
from sklearn.decomposition import PCA


embedders = {
    #"vgg19": {"data": Embedder_VGG19(reducer=PCA(n_components=50))},
    #"pose": {"data": Embedder_Poses()},
    #"face": {"data": Embedder_Face()},
    #"raw": {"data": Embedder_Raw(reducer=PCA(n_components=100))},
    #"clip": {"data": Embedder_CLIP()}
}

make_model(
    model_folder="models/model_name",
    embedders=embedders,
    data_root="/path/to/folder/or/csv" # CSV file or folder
    )
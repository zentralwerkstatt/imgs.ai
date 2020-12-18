from embedders import Embedder_Poses, Embedder_VGG19, Embedder_Raw, Embedder_Face
from train import make_model
from sklearn.decomposition import PCA


embedders = {
    "poses": Embedder_Poses(),
    "vgg19": Embedder_VGG19(reducer=PCA(n_components=50)),
    "raw": Embedder_Raw(reducer=PCA(n_components=50))
}

make_model(
    model_folder="models/model",
    embedders=embedders,
    data_root="data.csv" # Or folder
    )
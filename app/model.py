from app.util import set_cuda, load_img
import numpy as np
import os
import dill as pickle
import json
from annoy import AnnoyIndex
import h5py
from app.embedders import *
from sklearn.decomposition import PCA, IncrementalPCA # Used by dynamic imports
from app.index import Index
import typing


class EmbeddingModel:

    def __init__(self, model_folder:str):
        with open(os.path.join(model_folder, "config.json"), "r") as config: # Must exist
            self.config = json.load(config)
        self.model_folder = model_folder # Relative  
        self.model_name = os.path.basename(model_folder)

        # Neighborhoods
        self.anns = {}
        for emb_type, emb_type_props in self.config["emb_types"].items():
            self.anns[emb_type] = {}
            for metric in emb_type_props["metrics"]:
                self.anns[emb_type][metric] = AnnoyIndex(self.config["emb_types"][emb_type]["dims"], metric)
                self.anns[emb_type][metric].load(os.path.join(self.model_folder, f"{emb_type}_{metric}.ann")) # Nmaps the file

        # Metadata
        self.metadata = h5py.File(os.path.join(self.model_folder, "metadata.hdf5"), "r") # Must exist

    def __len__(self):
        return int(self.config["model_len"])

    def decode_metadata(self, metadata:str) -> list[str]:
        return json.loads(metadata.decode("utf-8"))

    def get_metadata(self, idx:Index) -> list[str]:
        return self.decode_metadata(self.metadata["metadata"][int(idx.idx)]) # Indices are strings
    
    def get_vectors_for_idx(self, idx:Index) -> dict[dict[np.ndarray]]:
        vectors = {}
        for emb_type, emb_type_props in self.config["emb_types"].items():
            vectors[emb_type] = {}
            for metric in emb_type_props["metrics"]:
                vectors[emb_type][metric] = self.anns[emb_type][metric].get_item_vector(int(idx.idx))  # Indices are strings
        return vectors

    def get_nns(self, emb_type:str, n:str, pos_idxs:list[Index], neg_idxs:list[Index], metric:str, search_k:int=-1) -> list[str]:
        nns = []

        queries = [str(idx) for idx in [pos_idxs + neg_idxs]] # Indices are strings

        # Convert Index-type objects to vectors
        pos_idxs = [idx.get_vectors(self, emb_type, metric) for idx in pos_idxs]
        neg_idxs = [idx.get_vectors(self, emb_type, metric) for idx in neg_idxs]

        n = int(n)

        # Do the math
        if pos_idxs or neg_idxs:

            if len(pos_idxs) > 0 and len(neg_idxs) > 0:  # Arithmetic
                vectors = np.array(pos_idxs + neg_idxs)
                centroid = vectors.mean(axis=0)
                pos_vectors = np.array(pos_idxs)
                neg_vectors = np.array(neg_idxs)

                pos_sum = 0
                for vector in pos_vectors:
                    pos_sum += vector
                centroid += pos_sum

                neg_sum = 0
                for vector in neg_vectors:
                    neg_sum += vector
                centroid -= neg_sum

                nns = self.anns[emb_type][metric].get_nns_by_vector(centroid, n, search_k=search_k, include_distances=False)

            elif len(pos_idxs) > 1 and len(neg_idxs) == 0: # Multiple positives
                vectors = np.array(pos_idxs)
                centroid = vectors.mean(axis=0)
                nns = self.anns[emb_type][metric].get_nns_by_vector(centroid, n, search_k=search_k, include_distances=False)

            elif len(pos_idxs) == 1:  # Single positive
                vector = np.array(pos_idxs[0])
                nns = self.anns[emb_type][metric].get_nns_by_vector(vector, n, search_k=search_k, include_distances=False)

            else: # Single or multiple negatives, return random
                nns = np.random.randint(len(self), size=n)

        else:  # Return random
            nns = np.random.randint(len(self), size=n)

        nns = [str(nn) for nn in nns] # Indices are strings

        # Remove queries
        nns = set(nns) - set(queries)

        return nns  

    def transform(self, paths:list[str]) -> dict[dict[np.ndarray]]:
        device = set_cuda()

        # Recreate embedders object from description only
        with open(os.path.join(self.model_folder, "embedders.pytxt")) as f: # Must exist
            embedders_string = f.read()
        locals = {}
        exec(embedders_string, globals(), locals)
        embedders = locals['embedders']
        
        # Replace reducers with fitted versions from file
        for emb_type, embedder in embedders.items():
            if embedder.reducer:
                with open(os.path.join(self.model_folder, f"{emb_type}_reducer.dill"), "rb") as f: # Must exist
                    embedders[emb_type].reducer = pickle.load(f)

        # Allocate space
        embs = {}
        for emb_type, embedder in embedders.items():            
            embs[emb_type] = np.zeros((len(paths), embedder.feature_length))

        # Extract embeddings
        for i, path in enumerate(paths):
            for emb_type, embedder in embedders.items():
                embs[emb_type][i] = embedder.transform(load_img(path), device)

        # Reduce if reducer given
        for emb_type, embedder in embedders.items():
            if embedder.reducer: embs[emb_type] = embedder.reducer.transform(embs[emb_type])

        # Save memory
        del embedders

        return embs
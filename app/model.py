from app.util import set_cuda, load_img
import numpy as np
import os
import dill as pickle
import json
from annoy import AnnoyIndex
import h5py
from app.embedders import *
from sklearn.decomposition import PCA, IncrementalPCA # Used by dynamic imports


class EmbeddingModel:

    def __init__(self, model_folder):
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

    def decode_metadata(self, metadata):
        return json.loads(metadata.decode("utf-8"))

    def get_metadata(self, idx):
        return self.decode_metadata(self.metadata["metadata"][int(idx)]) # Indices are strings
    
    def get_vectors_for_idx(self, idx):
        vectors = {}
        for emb_type, emb_type_props in self.config["emb_types"].items():
            vectors[emb_type] = {}
            for metric in emb_type_props["metrics"]:
                vectors[emb_type][metric] = self.anns[emb_type][metric].get_item_vector(int(idx))  # Indices are strings
        return vectors

    def get_nns_from_vectors(self, emb_type, n, pos_idxs, neg_idxs, metric, search_k=-1):
        nns = []

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

                nns = self.anns[emb_type][metric].get_nns_by_vector(centroid, int(n), search_k=search_k, include_distances=False)

            elif len(pos_idxs) > 1 and len(neg_idxs) == 0: # Multiple positives
                vectors = np.array(pos_idxs)
                centroid = vectors.mean(axis=0)
                nns = self.anns[emb_type][metric].get_nns_by_vector(centroid, int(n), search_k=search_k, include_distances=False)

            elif len(pos_idxs) == 1:  # Single positive
                vector = np.array(pos_idxs[0])
                nns = self.anns[emb_type][metric].get_nns_by_vector(vector, int(n), search_k=search_k, include_distances=False)

            else: # Single or multiple negatives, return random
                nns = np.random.randint(len(self), size=int(n))

        else:  # Return random
            nns = np.random.randint(len(self), size=int(n))

        # Do not remove queries from results as sanity check

        return [str(nn) for nn in nns]  # Indices are strings

    def transform(self, paths):
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
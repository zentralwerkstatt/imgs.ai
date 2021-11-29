from app.util import set_cuda, load_img, sort_dict, save_imgs_to
import numpy as np
import os
import dill as pickle
import json
from annoy import AnnoyIndex
import h5py
import csv
from app.embedders import *
from sklearn.decomposition import PCA, IncrementalPCA # Used by dynamic imports


class EmbeddingModel:

    def __init__(self):
        # Initialize with empty data structure
        self.model_folder = None
        
        self.metadata = {}
        self.paths = {}
        
        self.config = {
                "data_root": None,
                "model_len": None,
                "emb_types": {}
            }

    def __len__(self):
        return self.config["model_len"]

    def load(self, model_folder):
        self.model_folder = model_folder

        # Load configuration
        CONFIG_FPATH = os.path.join(model_folder, "config.json")
        with open(CONFIG_FPATH, "r") as config:
            self.config = json.load(config)

        # Read metadata and paths
        with open(os.path.join(self.model_folder, "metadata.csv")) as meta_file:
            for idx, row in enumerate(csv.reader(meta_file)):
                self.metadata[str(idx)] = row
                self.paths[str(idx)] = row[0]

    def extend(self, imgs, uploads_path):
        # Load uploads file
        uploads_file = os.path.join(self.model_folder, "uploads.hdf5")
        uploads = h5py.File(uploads_file, "a")  # Read/write/create

        paths, idxs = save_imgs_to(imgs, uploads_path)
        embs = self.transform(paths)

        for i, idx in enumerate(idxs):
            for emb_type in embs:
                uploads.create_dataset(f"{idx}/{emb_type}", compression="lzf", data=embs[emb_type][i])

        # Unload uploads file
        uploads.close()

        return idxs

    # For models where embeddings have been kept
    def get_features(self, emb_type):
        embs_file = os.path.join(self.model_folder, "embeddings.hdf5")
        embs = h5py.File(embs_file, "r")
        return embs[emb_type]

    def get_nns(self, emb_type, n, pos_idxs, neg_idxs, metric, vector=None, mode="ranking", search_k=-1, limit=None):
        # Load neighborhood file
        hood_file = os.path.join(self.model_folder, f"{emb_type}_{metric}.ann")
        ann = AnnoyIndex(self.config["emb_types"][emb_type]["dims"], metric)
        ann.load(hood_file)

        # Load uploads file
        uploads_file = os.path.join(self.model_folder, "uploads.hdf5")
        uploads = h5py.File(uploads_file, "a")

        # Get vectors from indices
        def vectors_from_idxs(idxs):
            vectors = []
            for idx in idxs:
                # Index for upload has UUID4 format
                if not idx.isnumeric():
                    vectors.append(uploads[idx][emb_type])
                else:
                    vectors.append(ann.get_item_vector(int(idx)))  # Indices are strings
                return vectors

        nns = []

        # Don't try to display more than we have
        n = min(n, len(self))

        # Get nearest neighbors
        if pos_idxs and neg_idxs:  # Arithmetic
            vectors = np.array(vectors_from_idxs(pos_idxs + neg_idxs))
            centroid = vectors.mean(axis=0)
            pos_vectors = np.array(vectors_from_idxs(pos_idxs))
            neg_vectors = np.array(vectors_from_idxs(neg_idxs))

            pos_sum = 0
            for vector in pos_vectors:
                pos_sum += vector
            centroid += pos_sum
            neg_sum = 0
            for vector in neg_vectors:
                neg_sum += vector
            centroid -= neg_sum
            nns = ann.get_nns_by_vector(
                centroid, n, search_k=search_k, include_distances=False
            )

        elif len(pos_idxs) > 1 and len(neg_idxs) == 0: # Ranking
            ranking = {}
            for idx in pos_idxs:
                vector = vectors_from_idxs([idx])[0]
                idx_nns, idx_scores = ann.get_nns_by_vector(
                    vector, n, search_k=search_k, include_distances=True
                )
                for nn, score in zip(idx_nns, idx_scores):
                    # If the neighbor was found already, just update the score
                    if nn in ranking:
                        if ranking[nn] > score:
                            ranking[nn] = score
                    else:
                        ranking[nn] = score
            nns = list(sort_dict(ranking).keys())

        else:  # Single
            if vector is None:
                vector = vectors_from_idxs(pos_idxs)
            nns = ann.get_nns_by_vector(
                vector[0], n, search_k=search_k, include_distances=False
            )

        # Unload neighborhood file
        ann.unload()

        nns = [str(nn) for nn in nns]  # Indices are strings
        nns = list(set(nns) - set(pos_idxs + neg_idxs))  # Remove queries
        nns = nns[:n]  # Limit to n

        return nns

    def transform(self, paths):
        device = set_cuda()

        # Recreate embedders object from description only
        with open(os.path.join(self.model_folder, "embedders.pytxt")) as f:
            embedders_string = f.read()
        locals = {}
        exec(embedders_string, globals(), locals)
        embedders = locals['embedders']
        
        # Replace reducers with fitted versions from file
        for emb_type, embedder in embedders.items():
            if embedder.reducer:
                with open(os.path.join(self.model_folder, f"{emb_type}_reducer.dill"), "rb") as f:
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
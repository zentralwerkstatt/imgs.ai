from util import new_dir, set_cuda, image_from_url, load_img
import h5py
from model import EmbeddingModel
from tqdm import tqdm
import os
import csv
from threading import Lock, Thread
from queue import Queue, Empty
import pickle
import json
import signal
import sys
import random
import numpy as np
from annoy import AnnoyIndex
from app import log


def collect_embed(X, embedders, data_root, num_workers, embs_file):
    device = set_cuda()

    # Allocate space
    log.info("Allocating space")
    embs = h5py.File(embs_file, "w")
    for emb_type, embedder in embedders.items():
        data = np.zeros((len(X), embedder['data'].feature_length))
        embs.create_dataset(emb_type.lower(), compression="lzf", data=data)

    # Set up threading
    pbar_success = tqdm(total=len(X), desc="Embedded")
    pbar_failure = tqdm(total=len(X), desc="Failed")
    q = Queue()
    l = Lock()
    valid_idxs = []

    # Catch interruptions to be able to close file
    def signal_handler(sig, frame):
        log.info("Shutting down gracefully...")
        embs.create_dataset("valid_idxs", compression="lzf", data=np.array(valid_idxs))
        embs.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Define and start queue
    def _worker():
        while True:
            try:
                i, x = q.get(timeout=5)
            except Empty:
                break
            path = x[0]
            success = False
            try:
                if path.startswith("http"):
                    img = image_from_url(path)
                else:
                    img = load_img(os.path.join(data_root, path))
            except:
                img = None
            if img:
                with l:
                    for emb_type, embedder in embedders.items():
                        embs[emb_type.lower()][i] = embedder['data'].transform(img, device)
                    valid_idxs.append(i)
                    success = True
            with l:
                if success:
                    pbar_success.update(1)
                else:
                    pbar_failure.update(1)
            q.task_done()

    for i in range(num_workers):
        t = Thread(target=_worker)
        t.daemon = True
        t.start()

    for i, x in enumerate(X):
        q.put((i, x))

    # Cleanup
    q.join()
    pbar_success.close()
    pbar_failure.close()
    embs.create_dataset("valid_idxs", compression="lzf", data=np.array(valid_idxs))
    embs.close()


def train(X, model_folder, embedders, data_root, num_workers, metrics=["angular", "euclidean", "manhattan"], n_trees=100):
    log.info(f'Checking {model_folder}')
    new_dir(model_folder)

    # Set up
    log.info(f'Setting up config')
    config = {}
    config["data_root"] = data_root
    config["metrics"] = metrics

    # Create or load raw embeddings
    embs_file = os.path.join(model_folder, "embeddings.hdf5")
    if not os.path.isfile(embs_file):
        collect_embed(X, embedders, data_root, num_workers, embs_file)

    embs = h5py.File(embs_file, "r")
    log.debug(embs_file, embs)
    valid_idxs = list(embs["valid_idxs"])
    config["embs_file"] = "embeddings.hdf5"
    config["model_len"] = len(valid_idxs)

    # Allocate cache
    log.info(f'Allocating cache')
    cache_file = os.path.join(model_folder, "cache.hdf5")
    cache = h5py.File(cache_file, "w")

    # Reduce if reducer given
    log.info(f'Applying dimensionality reduction')
    for emb_type, embedder in embedders.items():
        data = embs[emb_type.lower()]
        if embedder['data'].reducer:
            data = embedder['data'].reducer.fit_transform(embs[emb_type.lower()])
        cache.create_dataset(emb_type.lower(), data=data, compression="lzf")

    # Build and save neighborhoods
    log.info(f'Building neighborhoods')
    config["hood_files"] = {}
    for emb_type, embedder in embedders.items():
        config["hood_files"][emb_type.lower()] = {}
        for metric in metrics:
            if embedder['data'].reducer:
                dims = embedder['data'].reducer.n_components
            else:
                dims = embedder['data'].feature_length
            ann = AnnoyIndex(dims, metric)
            for i, idx in enumerate(valid_idxs):
                ann.add_item(i, cache[emb_type.lower()][idx])
            ann.build(n_trees)
            hood_file = os.path.join(model_folder, f"{emb_type.lower()}_{metric}.ann")
            ann.save(hood_file)
            config["hood_files"][emb_type.lower()][metric] = f"{emb_type.lower()}_{metric}.ann"

    # Align and write metadata
    log.info(f'Aligning metadata')
    meta = []
    for idx in valid_idxs:
        meta.append(X[idx])
    meta_file = os.path.join(model_folder, "metadata.csv")
    csv.writer(open(meta_file, "w")).writerows(meta)
    config["meta_file"] = "metadata.csv"

    # Save fitted embedders
    log.info("Writing additional data")
    for emb_type, embedder in embedders.items():
        embedder['data'].model = None  # Delete models to save memory
    embedders_file = os.path.join(model_folder, "embedders.pickle")
    with open(embedders_file, "wb") as f:
        pickle.dump(embedders, f)
    config["embedders_file"] = "embedders.pickle"

    # More config
    config["dims"] = {}
    config["emb_types"] = []
    for emb_type, embedder in embedders.items():
        config["dims"][emb_type.lower()] = {}
        config["emb_types"].append(emb_type.lower())
        if embedder['data'].reducer:
            dims = embedder['data'].reducer.n_components
        else:
            dims = embedder['data'].feature_length
        config["dims"][emb_type.lower()] = dims

    # Save config
    config_file = os.path.join(model_folder, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f)

    # Cleanup
    embs.close()
    cache.close()
    os.remove(cache_file)


def arrange_data(X, shuffle=0, max_data=0):
    log.info('Arranging data')
    if shuffle: random.shuffle(X)
    if max_data: X = X[:max_data]
    return X


def make_model(model_folder, embedders, data_root, num_workers=64, shuffle=False, max_data=None):
    X = []
    
    if data_root.endswith(".csv"):
        with open(data_root, "r") as f:
            meta = csv.reader(f)
            log.debug(meta)
            for row in meta:
                if len(row) == 1: X.append(row); continue
                fname = row[0]
                url = row[1]
                X.append([fname, url] + [field for field in row[2:]])
        
        X = arrange_data(X, shuffle, max_data)

        log.info('Start training')
        train(X=X, data_root=None,
                model_folder=model_folder,
                embedders=embedders,
                num_workers=num_workers
            )

    else: # Not CSV
        for root, _, files in os.walk(data_root):
            for fname in files:
                X.append([os.path.relpath(os.path.join(root, fname), start=data_root), "", None])
        
        X = arrange_data(X, shuffle, max_data)

        log.info('Training')
        train(
            X=X,
            data_root=data_root,
            model_folder=model_folder,
            embedders=embedders,
            num_workers=num_workers,
        )

    log.info('Done')
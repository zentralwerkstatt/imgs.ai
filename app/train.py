# Filter annyoing PyTorch warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import os
import h5py
from tqdm import tqdm
import csv
from threading import Lock, Thread
from queue import Queue, Empty
import dill as pickle # https://discuss.pytorch.org/t/cant-pickle-local-object-dataloader-init-locals-lambda/31857/32
import json
from annoy import AnnoyIndex
import logging
from util import load_img, img_from_url, set_cuda


# Logging
logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(name)s : %(message)s")
log.info("Succesfully set up logging")


def collect_embed(X, embedders, data_root, num_workers, embs_file, start, end):
    device = set_cuda()

    X_dict = {}
    for i, x in enumerate(X):
        X_dict[i] = x

    if end is None:
        end = len(X_dict)
    
    # Allocate space
    log.info("Allocating space")
    embs = h5py.File(embs_file, "a")
    valid_idxs = []
    if start == 0:
        for emb_type, embedder in embedders.items():
            embs.create_dataset(emb_type.lower(), 
                                compression="lzf", 
                                shape=(len(X), embedder.feature_length))
    else:
        valid_idxs.extend(list(embs["valid_idxs"][:]))
        del embs["valid_idxs"]

    # Set up threading
    pbar_success = tqdm(total=(end-start), desc="Embedded")
    pbar_failure = tqdm(total=(end-start), desc="Failed")
    q = Queue()
    l = Lock()
    
    # Define and start queue
    def _worker():
        while True:
            try:
                i, x = q.get()
            except Empty:
                break
            path = x[0]
            success = False
            try:
                if path.startswith("http"):
                    img = img_from_url(path)
                else:
                    img = load_img(os.path.join(data_root, path))
            except:
                img = None
            if img:
                with l:
                    for emb_type, embedder in embedders.items():
                        embs[emb_type.lower()][i] = embedder.transform(img, device)
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

    for i,x in X_dict.items():
        if i>=start and i<=end:
            q.put((i, x))

    # Cleanup
    q.join()
    pbar_success.close()
    pbar_failure.close()
    embs.create_dataset("valid_idxs", compression="lzf", data=np.array(valid_idxs))
    embs.close()


def train(X, model_folder, embedders, data_root, num_workers=32, start=0, end=None, n_trees=100, build=True):
    # Set up
    log.info(f'Setting up config')
    config = {}

    # Create or load raw embeddings
    embs_file = os.path.join(model_folder, "embeddings.hdf5")
    collect_embed(X, embedders, data_root, num_workers, embs_file, start, end)

    if build:

        embs = h5py.File(embs_file, "r")
        valid_idxs = list(embs["valid_idxs"])
        config["model_len"] = len(valid_idxs)

        # Allocate cache
        log.info(f'Allocating cache')
        cache_file = os.path.join(model_folder, "cache.hdf5")
        cache = h5py.File(cache_file, "w")

        # Reduce if reducer given
        log.info(f'Applying dimensionality reduction')
        for emb_type, embedder in embedders.items():
            data = None
            if embedder.reducer:
                log.info(embedder.reducer)
                data = embedder.reducer.fit_transform(embs[emb_type.lower()]) 
            else:
                data = embs[emb_type.lower()]
            cache.create_dataset(emb_type.lower(), data=data, compression="lzf")

        # Build and save neighborhoods
        log.info(f'Building neighborhoods')
        config["emb_types"] = {}
        for emb_type, embedder in embedders.items():
            config["emb_types"][emb_type.lower()] = {}
            config["emb_types"][emb_type.lower()]["metrics"] = []
            for metric in embedder.metrics:
                config["emb_types"][emb_type.lower()]["metrics"].append(metric)
                if embedder.reducer:
                    dims = embedder.reducer.n_components
                else:
                    dims = embedder.feature_length
                config["emb_types"][emb_type.lower()]["dims"] = dims
                ann = AnnoyIndex(dims, metric)
                for i, idx in enumerate(valid_idxs):
                    ann.add_item(i, cache[emb_type.lower()][idx])
                ann.build(n_trees)
                hood_file = os.path.join(model_folder, f"{emb_type.lower()}_{metric}.ann")
                ann.save(hood_file)

        # Align and write metadata
        log.info(f'Aligning metadata')
        meta = []
        for idx in valid_idxs:
            meta.append(X[idx])
        meta_file = os.path.join(model_folder, "metadata.csv")
        csv.writer(open(meta_file, "w")).writerows(meta)

        # Save fitted reducers
        log.info("Saving fitted reducers")
        for emb_type, embedder in embedders.items():
            if embedder.reducer:
                reducer_file = os.path.join(model_folder, f"{emb_type}_reducer.dill")
                with open(reducer_file, "wb") as f:
                    pickle.dump(embedder.reducer, f)

        # Save config
        config_file = os.path.join(model_folder, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f)

        # Cleanup
        embs.close()
        cache.close()
        os.remove(cache_file)
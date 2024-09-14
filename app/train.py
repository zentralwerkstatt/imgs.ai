# Filter annyoing PyTorch warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

from embedders import Embedder_CLIP_ViT, Embedder_Poses, Embedder_Raw, Embedder_VGG19 # Dynamic imports
from sklearn.decomposition import IncrementalPCA # Dynamic imports
import numpy as np
import os
import h5py
from tqdm import tqdm
from threading import Lock, Thread
from queue import Queue, Empty
import dill as pickle # https://discuss.pytorch.org/t/cant-pickle-local-object-dataloader-init-locals-lambda/31857/32
import json
from annoy import AnnoyIndex
import logging
from util import load_img, img_from_url, set_cuda
import csv


# Logging
logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s : %(message)s")

def collect_embed(model_folder, num_workers=32, max_data=None):
    device = set_cuda()

    assert os.path.exists(model_folder), "Model folder does not exist"
    assert os.path.exists(os.path.join(model_folder, "metadata.csv")), "metadata.csv does not exist"
    assert os.path.exists(os.path.join(model_folder, "embedders.pytxt")), "embedders.pytxt does not exist"

    # Read original metadata from CSV
    with open(os.path.join(model_folder, "metadata.csv")) as f:
        X = [row for row in csv.reader(f)]
    if max_data: 
        X = X[:max_data]
    
    # Recreate embedders object from description only
    with open(os.path.join(model_folder, "embedders.pytxt")) as f:
        embedders_string = f.read()
    locals = {}
    exec(embedders_string, globals(), locals)
    embedders = locals['embedders']
    
    # Allocate space
    log.info("Allocating space")
    embs_file = os.path.join(model_folder, "embeddings.hdf5")
    embs = h5py.File(embs_file, "a")
    valid_idxs = []
    for emb_type, embedder in embedders.items():
        embs.create_dataset(emb_type.lower(), 
                            compression="lzf", 
                            shape=(len(X), embedder.feature_length))

    # Set up threading
    pbar_success = tqdm(total=len(X), desc="Embedded")
    pbar_failure = tqdm(total=len(X), desc="Failed")
    q = Queue()
    l = Lock()
    
    # Define and start queue
    def _worker():
        while True:
            try:
                i, x = q.get()
            except Empty:
                break
            path = x[0] # First column is path relative to data folder
            success = False
            try:
                if path.startswith("http"):
                    img = img_from_url(path)
                else:
                    img = load_img(os.path.join(model_folder, "data", path))
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

    for i, x in enumerate(X):
        q.put((i, x))

    # Cleanup
    q.join()
    pbar_success.close()
    pbar_failure.close()
    log.info("Writing embeddings")
    embs.create_dataset("valid_idxs", compression="lzf", data=np.array(valid_idxs))
    embs.close()

def build(model_folder, n_trees=100, private=False):
    config = {}

    assert os.path.exists(model_folder), "Model folder does not exist"
    assert os.path.exists(os.path.join(model_folder, "metadata.csv")), "metadata.csv does not exist"
    assert os.path.exists(os.path.join(model_folder, "embedders.pytxt")), "embedders.pytxt does not exist"
    assert os.path.exists(os.path.join(model_folder, "embeddings.hdf5")), "embeddings.hdf5 does not exist"

    # Read original metadata from CSV
    with open(os.path.join(model_folder, "metadata.csv")) as f:
        X = [row for row in csv.reader(f)]

    # Recreate embedders object from description only
    with open(os.path.join(model_folder, "embedders.pytxt")) as f:
        embedders_string = f.read()
    locals = {}
    exec(embedders_string, globals(), locals)
    embedders = locals['embedders']

    # Load raw embeddings
    log.info(f'Loading embeddings')
    embs_file = os.path.join(model_folder, "embeddings.hdf5")
    embs = h5py.File(embs_file, "r")
    valid_idxs = list(embs["valid_idxs"])
    config["private"] = private
    config["model_len"] = len(valid_idxs)

    # Allocate cache
    log.info(f'Allocating cache')
    cache_file = os.path.join(model_folder, "cache.hdf5")
    cache = h5py.File(cache_file, "a")

    # Reduce if reducer given
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

    # Save fitted reducers
    log.info("Saving reducers")
    for emb_type, embedder in embedders.items():
        if embedder.reducer:
            reducer_file = os.path.join(model_folder, f"{emb_type}_reducer.dill")
            with open(reducer_file, "wb") as f:
                pickle.dump(embedder.reducer, f)

    # Save config
    config_file = os.path.join(model_folder, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f)

    # Write metadata
    log.info(f'Aligning metadata')    
    meta_file = os.path.join(model_folder, "metadata.hdf5")
    meta = h5py.File(meta_file, "a")
    dtype = h5py.string_dtype(encoding='utf-8')
    meta.create_dataset("metadata", (len(X),), dtype=dtype, compression="lzf")
    for idx, x in enumerate(X):
        meta["metadata"][idx] = json.dumps(x) # JSON gives us strings

    # Cleanup
    embs.close()
    cache.close()
    meta.close()
    os.remove(cache_file)
    log.info("All done")
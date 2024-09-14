from embedders import Embedder_CLIP_ViT, Embedder_Poses, Embedder_Raw, Embedder_VGG19 # Dynamic imports
from sklearn.decomposition import IncrementalPCA # Dynamic imports
import csv
import shutil
import os
from uuid import uuid4
from tqdm import tqdm
from threading import Lock, Thread
from queue import Queue, Empty
from util import img_from_url, new_dir
from train import build
import h5py
import numpy as np


old_model = "static/models/Rijksmuseum" # Name of the existing model
new_model = "static/models/Rijksmuseum_1K_local" # Name of the local copy
resize = 640
max_data = 1000
num_workers = 32

new_dir(f"{new_model}/data") # Creates both model folder and data folder

shutil.copy(f"{old_model}/embeddings.hdf5", f"{new_model}/embeddings.hdf5") # Overwritten if exists
shutil.copy(f"{old_model}/embedders.pytxt", f"{new_model}/embedders.pytxt") # Overwritten if exists
shutil.copy(f"{old_model}/metadata.csv", f"{new_model}/metadata.old")

# Read original metadata from CSV
with open(f"{new_model}/metadata.old", "r") as f:
    X = [row for row in csv.reader(f)]

# Revise metadata
X_revised = []
for x in X:
    X_revised.append([f"{str(uuid4())}.jpg"] + x[1:])
X_revised

# Write revised metadata to CSV
with open(f"{new_model}/metadata.csv", "w") as f:
    csv.writer(f).writerows(X_revised)

# Truncate (on the embeddings side, not the metadata side)
# This means we create filenames for files we do not download
X_collect = X_revised
if max_data:
    embs_file = os.path.join(f"{new_model}/embeddings.hdf5")
    embs = h5py.File(embs_file, "a")
    valid_idxs = list(embs["valid_idxs"])
    # Select the first max_data valid indices
    truncated_idxs = valid_idxs[:max_data]
    # Truncate data
    X_collect = [X[i] for i in truncated_idxs]
    # Truncate embeddings
    del embs["valid_idxs"]
    embs.create_dataset("valid_idxs", compression="lzf", data=np.array(truncated_idxs))
    embs.close()

# Set up threading
pbar_success = tqdm(total=len(X_collect), desc="Downloaded")
pbar_failure = tqdm(total=len(X_collect), desc="Failed")
q = Queue()
l = Lock()

# TODO: This should be a util function that takes a function as a parameter
# Define and start queue
def _worker():
    while True:
        try:
            i, x, x_r = q.get()
        except Empty:
            break
        path = x[0]
        uuid = x_r[0]
        try:
            img = img_from_url(path)
            if resize:
                img.thumbnail((resize, resize))
            img.save(f"{new_model}/data/{uuid}")
            success = True
        except:
            success = False
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

for i in range(len(X_collect)):
    x = X[i]
    x_r = X_revised[i]
    q.put((i, x, x_r))

# Cleanup
q.join()
pbar_success.close()
pbar_failure.close()

# Rebuild
build(new_model)
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


old_model = "Rijksmuseum" # Name of the existing model
new_model = "Rijksmuseum_10K_local" # Name of the local copy
model_dir = f"/Users/fabian/Desktop/dev/imgs.ai/app/static/models"  # Absolute, storage for models
num_workers = 32
resize = 640
max_data = 10000

new_model_data_root = f"{model_dir}/{new_model}/data" # Where to download files to
new_dir(new_model_data_root) # Creates both model folder and data folder

print(f"Converting {model_dir}/{old_model} to {model_dir}/{new_model}")
assert os.path.exists(f"{model_dir}/{old_model}"), "Model does not exist"

shutil.copy(f"{model_dir}/{old_model}/embeddings.hdf5", f"{model_dir}/{new_model}/embeddings.hdf5") # Overwritten if exists
shutil.copy(f"{model_dir}/{old_model}/embedders.pytxt", f"{model_dir}/{new_model}/embedders.pytxt") # Overwritten if exists
shutil.copy(f"{model_dir}/{old_model}/metadata.csv", f"{model_dir}/{new_model}/metadata.old")

X = []

# Read data from CSV
with open(f"{model_dir}/{new_model}/metadata.old", "r") as f:
    meta = csv.reader(f)
    for row in meta:
        X.append(row)

# Truncate
if max_data:
    X = X[:max_data]

# Generate new filenames
# Keep order intact – otherwise we can't match to idx
new_meta = []
for i, x in enumerate(X):
    fname = f"{new_model}_{str(uuid4())}.jpg"
    new_meta.append([fname] + x[1:])

# Write
with open(f"{model_dir}/{new_model}/metadata.csv", "w") as f:
    csv.writer(f).writerows(new_meta)

# Set up threading
pbar_success = tqdm(total=len(X), desc="Downloaded")
pbar_failure = tqdm(total=len(X), desc="Failed")
q = Queue()
l = Lock()

# Define and start queue
def _worker():
    while True:
        try:
            i, x, fname = q.get()
        except Empty:
            break
        path = x[0]
        try:
            img = img_from_url(path)
            if resize:
                img.thumbnail((resize, resize))
            img.save(os.path.join(new_model_data_root, fname))
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

for i, x in enumerate(X):
    q.put((i, x, new_meta[i][0]))

# Cleanup
q.join()
pbar_success.close()
pbar_failure.close()

# Rebuild
with open(f"{model_dir}/{new_model}/embedders.pytxt") as f:
    embedders_string = f.read()
    locals = {}
    exec(embedders_string, globals(), locals)
    embedders = locals['embedders']
build(X, f"{model_dir}/{new_model}", embedders)
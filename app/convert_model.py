# Convert a remote into a local model
import csv
import shutil
import os
from uuid import uuid4
from tqdm import tqdm
from threading import Lock, Thread
from queue import Queue, Empty
from util import img_from_url, new_dir
import pathlib


old_model = "public/Rijksmuseum" # Name of the existing model
new_model = "private/Rijksmuseum_Local" # Name of the local copy
num_workers = 32
max_data = 100 # Limit to max_data images (useful for testing purposes)
resize = 640

new_model_name = new_model.split("/")[-1]
grandparent_dir = pathlib.Path(__file__).resolve(strict=True).parents[1]
model_dir = f"{grandparent_dir}/models"
new_model_data_root = f"{grandparent_dir}/app/static/data/{new_model_name}" # Where to download files to

print(f"Converting {model_dir}/{old_model} to {model_dir}/{new_model}")
assert os.path.exists(f"{model_dir}/{old_model}"), "Model does not exist"
assert os.path.exists(f"{model_dir}/{old_model}/metadata.csv"), "Metadata does not exist"
assert not os.path.exists(f"{model_dir}/{new_model}"), "New model already exists"
assert not os.path.exists(new_model_data_root), "New data root already exists"

shutil.copytree(f"{model_dir}/{old_model}", f"{model_dir}/{new_model}")
shutil.move(f"{model_dir}/{new_model}/metadata.csv", f"{model_dir}/{new_model}/metadata.old")
new_dir(new_model_data_root)

X = []

# Read
with open(f"{model_dir}/{new_model}/metadata.old", "r") as f:
    meta = csv.reader(f)
    for row in meta:
        X.append(row)

X = X[:max_data]

# Generate new filenames
# Keep order intact – otherwise we can't match to idx
new_meta = []
X_dict = {}
for i, x in enumerate(X):
    X_dict[i] = x
    fname = f"{new_model_name}_{str(uuid4())}.jpg"
    new_meta.append([fname] + x[1:])

# Write
with open(f"{model_dir}/{new_model}/metadata.csv", "w") as f:
    csv.writer(f).writerows(new_meta)

# Set up threading
pbar_success = tqdm(total=len(X_dict), desc="Downloaded")
pbar_failure = tqdm(total=len(X_dict), desc="Failed")
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

for i,x in X_dict.items():
    q.put((i, x, new_meta[i][0]))

# Cleanup
q.join()
pbar_success.close()
pbar_failure.close()
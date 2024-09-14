import h5py
import os
import csv
from tqdm import tqdm
import json


model_folder = "static/models/Rijksmuseum"

with open(os.path.join(model_folder, "metadata.csv")) as f:
    X = [row for row in csv.reader(f)]

meta_file = os.path.join(model_folder, "metadata.hdf5")
meta = h5py.File(meta_file, "a")
dtype = h5py.string_dtype(encoding='utf-8')
meta.create_dataset("metadata", (len(X),), dtype=dtype, compression="lzf")

for idx, x in enumerate(tqdm(X, total=len(X))):
    meta["metadata"][idx] = json.dumps(x) # JSON gives us strings

meta.close()
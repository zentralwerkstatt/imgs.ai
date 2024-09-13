import h5py
import os
import csv
import random
from tqdm import tqdm
import json


def lencode(row):
    return json.dumps(row)

def ldecode(string):
    return json.loads(string)

model_folder = "static/models/Rijksmuseum"

meta_file = os.path.join(model_folder, "metadata.csv")
old_meta = csv.reader(open(meta_file))
row_count = sum(1 for row in old_meta)
old_meta = csv.reader(open(meta_file)) # Open again to go back to start

new_meta_file = os.path.join(model_folder, "metadata.hdf5")
new_meta = h5py.File(new_meta_file, "a")
dtype = h5py.string_dtype(encoding='utf-8')
new_meta.create_dataset("metadata", (row_count,), dtype=dtype, compression="lzf")

for idx, row in enumerate(tqdm(old_meta, total=row_count)):
    new_meta["metadata"][idx] = lencode(row)

new_meta.close()

# Test
new_meta = h5py.File(new_meta_file, "r")
k = random.choice(range(row_count))
print(ldecode(new_meta["metadata"][k].decode("utf-8")))
new_meta.close()
import numpy as np
import PIL.Image
import torch as t
from uuid import uuid4
import os
from random import sample
import h5py
from io import BytesIO
import requests
import time
import random
from flask import send_file
from io import StringIO
import clip
import pathlib


def set_cuda():
    device = "cuda" if t.cuda.is_available() else "cpu"
    return device


device = set_cuda()
# print(clip.available_models())
model_vit, _ = clip.load("ViT-B/32", device=device)
model_rn, transforms_rn = clip.load("RN50", device=device)


def get_current_dir():
    return pathlib.Path(__file__).parent.absolute()


def get_parent_dir(dir):
    return pathlib.Path(dir).parent.absolute()


def CLIP_text(text): # Use RN50 if not indicated otherwise
    with t.no_grad():
        text_features = model_vit.encode_text(clip.tokenize(text).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def img_from_url(url, max_tries=10):
    tries = 0
    while tries < max_tries:
        try:
            response = requests.get(url, timeout=30)
            img_bytes = BytesIO(response.content)
            img = PIL.Image.open(img_bytes).convert("RGB")
            return img
        except:
            tries += 1
        time.sleep(1)


def save_imgs_to(imgs, prefix, folder):    
    new_dir(folder)
    paths = []
    idxs = []
    for img in imgs:
        if isinstance(img, str): # URL or file path
            if img.startswith("http"):
                img = img_from_url(img)
            else:
                img = PIL.Image.open(img).convert("RGB")
        else: # Data stream
            stream = BytesIO(img.read())
            img = PIL.Image.open(stream).convert("RGB")
        idx = f"{prefix}_{str(uuid4())}.jpg"
        path = str(os.path.join(folder, idx))
        img.save(path)
        paths.append(path)
        idxs.append(idx)
    return paths, idxs


def sample_range(n, k):
    return sample(list(range(n)), k=k)


def serve_pil_image(img):
    img_io = StringIO()
    img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def load_img(path):
    return PIL.Image.open(path).convert("RGB")


def sort_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


def from_device(tensor):
    return tensor.detach().cpu().numpy()


def new_dir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder

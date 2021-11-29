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
from matplotlib import cm
import clip
from torchray.attribution.grad_cam import grad_cam # inspired by: https://github.com/HendrikStrobelt/miniClip
from math import floor


def set_cuda():
    device = "cuda" if t.cuda.is_available() else "cpu"
    return device


device = set_cuda()
# print(clip.available_models())
model_vit, _ = clip.load("ViT-B/32", device=device)
model_rn, transforms_rn = clip.load("RN50", device=device)


def min_max_norm(array):
    lim = [array.min(), array.max()]
    array = array - lim[0] 
    array.mul_(1 / (1.e-10+ (lim[1] - lim[0])))
    # array = torch.clamp(array, min=0, max=1)
    return array


def heatmap(original, saliency, alpha):
    hm = cm.afmhot(min_max_norm(saliency).numpy()) # Get heatmap
    hm = PIL.Image.fromarray((hm*255.).astype(np.uint8)).convert("RGB") # Convert to image

    h = original.height
    w = original.width
    min_dim = min(w, h)
    max_dim = max(w, h)
    border = floor((max_dim - min_dim)/2)

    # Scale up square heatmap to smaller image dimension
    hm = hm.resize((min_dim, min_dim))

    original = np.array(original)
    hm = np.array(hm)
    canvas = np.zeros_like(original)

    # Paste heatmap onto empty canvas of original image size
    if w == min_dim:
        canvas[border:border+min_dim,0:min_dim,:] = hm
    else:
        canvas[0:min_dim,border:border+min_dim,:] = hm

    # Will it blend?
    return PIL.Image.fromarray((alpha * canvas + (1-alpha) * original).astype(np.uint8))


def CLIP_text(text): # Use RN50 if not indicated otherwise
    with t.no_grad():
        text_features = model_vit.encode_text(clip.tokenize(text).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def CLIP_gradcam(text, img_path):
    text = clip.tokenize(text).to(device)
    original = load_img(img_path)
    img = transforms_rn(original).unsqueeze(0).to(device)
    
    with t.no_grad():
        image_features = model_rn.encode_image(img)
        text_features = model_rn.encode_text(text)
        image_features_norm = image_features.norm(dim=-1, keepdim=True)
        image_features_new = image_features / image_features_norm
        text_features_norm = text_features.norm(dim=-1, keepdim=True)
        text_features_new = text_features / text_features_norm

    text_prediction = (text_features_new * image_features_norm)
    saliency = grad_cam(model_rn.visual, img.type(model_rn.dtype), text_prediction, saliency_layer="layer4.2.relu")

    return heatmap(original, saliency[0][0,].detach().type(t.float32).cpu(), alpha=0.7)


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


def arrange_data(X, shuffle=0, max_data=0):
    if shuffle: random.shuffle(X)
    if max_data: X = X[:max_data]
    return X


def save_imgs_to(imgs, folder):    
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
        idx = str(uuid4())
        path = str(os.path.join(folder, f"{idx}.jpg"))
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

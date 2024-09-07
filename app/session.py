import os
from app.util import sample_range, CLIP_text, new_dir
from app import app, models, log, Config
from flask import url_for


# Per-user state, deals with server-side models and serialization as client session
class Session:

    size = Config.DEFAULT_SIZE
    n = Config.DEFAULT_N
    clip_prompt = ""
    public = True

    def __init__(self, flask_session):
        if "model" in flask_session:
            self.restore(flask_session)
        else:
            self.load_model(Config.DEFAULT_MODEL)
            self.emb_type = Config.DEFAULT_EMB_TYPE
            
    def store(self, flask_session):
        flask_session["model"] = self.model
        flask_session["model_len"] = self.model_len
        flask_session["size"] = self.size
        flask_session["emb_types"] = self.emb_types
        flask_session["emb_type"] = self.emb_type
        flask_session["metrics"] = self.metrics
        flask_session["metric"] = self.metric
        flask_session["res_idxs"] = self.res_idxs
        flask_session["pos_idxs"] = self.pos_idxs
        flask_session["neg_idxs"] = self.neg_idxs
        flask_session["n"] = self.n
        flask_session["clip_prompt"] = self.clip_prompt
        flask_session["public"] = self.public

    def restore(self, flask_session):
        self.model = flask_session["model"]
        self.model_len = flask_session["model_len"]
        self.size = flask_session["size"]
        self.emb_types = flask_session["emb_types"]
        self.emb_type = flask_session["emb_type"]
        self.metrics = flask_session["metrics"]
        self.metric = flask_session["metric"]
        self.res_idxs = flask_session["res_idxs"]
        self.pos_idxs = flask_session["pos_idxs"]
        self.neg_idxs = flask_session["neg_idxs"]
        self.n = flask_session["n"]
        self.clip_prompt = flask_session["clip_prompt"]
        self.public = flask_session["public"]

    def load_model(self, model, pin_idxs=None):
        files = []
        if pin_idxs:
            for idx in pin_idxs:
                files.append(self.get_path(idx))

        self.model = model
        self.model_len = models[self.model].config["model_len"]
        self.emb_types = list(models[self.model].config["emb_types"].keys())
        self.emb_type = Config.DEFAULT_EMB_TYPE
        self.metrics = models[self.model].config["emb_types"][self.emb_type]["metrics"]
        self.metric = self.metrics[0]
        self.res_idxs = []
        self.pos_idxs = []
        self.neg_idxs = []

        if files: self.extend(files)

    def get_nns(self):
        # If we have positive or mixed queries or CLIP prompt, search nearest neighbors
        if self.pos_idxs or self.clip_prompt:
            if self.clip_prompt:
                vector=CLIP_text(self.clip_prompt)
            else:
                vector= None
            self.res_idxs = models[self.model].get_nns(
                emb_type=self.emb_type,
                n=int(self.n),
                pos_idxs=self.pos_idxs,
                neg_idxs=self.neg_idxs,
                vector=vector,
                metric=self.metric,
                )
        # Else display random data points
        # (Negatives only filtered by routes.py)
        else:
            if self.res_idxs: # Keep randomized selection
                pass
            else:
                k = int(self.n)
                if k > self.model_len:
                    idxs = sample_range(self.model_len, self.model_len)
                else:
                    idxs = sample_range(self.model_len, k)
                self.res_idxs = [str(idx) for idx in idxs]  # Indices are strings
    
    def get_path(self, idx):
        if idx.startswith("upload"): # Uploaded file
            return f"{Config.DATA_PATH}/{self.model}/{idx}"
        else:
            path = models[self.model].paths[idx]
            if path.startswith("http"): 
                return path
            else: # Local data
                return os.path.join(Config.DATA_PATH, self.model, path)
            
    def get_url(self, idx):
        if idx.startswith("upload"): # Uploaded file
            return url_for("static", filename=f"data/{self.model}/{idx}")
        else:
            path = models[self.model].paths[idx]
            if path.startswith("http"):
                return path
            else: # Local data
                return url_for("static", filename=f"data/{self.model}/{path}")

    def get_metadata(self, idx):
        return models[self.model].metadata[idx]

    def extend(self, files):
        # Check if model has a folder in DATA_PATH
        model_data_path = os.path.join(Config.DATA_PATH, self.model)
        if not os.path.isdir(model_data_path):
            new_dir(model_data_path)
        self.pos_idxs += models[self.model].extend(files, model_data_path)
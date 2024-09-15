from app import models, jar
from app.index import ModelIndex, PromptIndex, UploadIndex
from PIL import Image

        
class Session:

    config = {
        "private": False,
    }
    data = {
        "pos_idxs":[],
        "neg_idxs":[],
        "res_idxs":[]
    }

    def __init__(self, flask_session):
        if "config" in flask_session and "data_id" in flask_session: # We have stored this before
            self.restore(flask_session)
        else:
            self.load_model("Rijksmuseum") # Default model
                        
    def store(self, flask_session):
        flask_session["config"] = self.config # Store config in a cookie
        flask_session["data_id"] = jar.store(self.data) # Store data on the server

    def restore(self, flask_session):
        self.config = flask_session["config"]
        self.data = jar.restore(flask_session["data_id"])

    def load_model(self, model_name):
        self.config["model_name"] = model_name
        self.config["model_names"] = [model_name for model_name in models.keys() if models[model_name].config["private"] == self.config["private"]]
        self.config["model_len"] = len(models[self.config["model_name"]])
        self.config["emb_types"] = list(models[self.config["model_name"]].config["emb_types"].keys())
        self.config["emb_type"] = self.config["emb_types"][0] # ALways put CLIP first in config.json
        self.config["metrics"] = models[self.config["model_name"]].config["emb_types"][self.config["emb_type"]]["metrics"]
        self.config["metric"] = self.config["metrics"][0]
        self.config["ns"] = ["25", "50", "75", "100"]
        self.config["n"] = "50"
        self.data["pos_idxs"] = [idx.keep() for idx in self.data["pos_idxs"]] # Keep from previous model
        self.data["neg_idxs"] = [idx.keep() for idx in self.data["neg_idxs"]] # Keep from previous model
        self.data["res_idxs"] = []

    def edit_config(self, field, value):
        if field == "emb_type":
            self.config["emb_type"] = value
            self.config["metrics"] = models[self.config["model_name"]].config["emb_types"][self.config["emb_type"]]["metrics"]
            self.config["metric"] = self.config["metrics"][0]
        elif field == "metric":
            self.config["metric"] = value
        elif field == "n":
            self.config["n"] = value
        
    def edit_idxs(self, action, idxs=None, upload=None, prompt=None):
        assert action in ["add_pos", "add_neg", "remove", "clear", "add_prompt", "add_upload"], f"Invalid action: {action}"
        
        # The form on the client side cannot handle Index-type objects and sends strings so we need to convert them back
        if (action == "add_pos" or action == "add_neg" or action=="remove") and idxs:
            session_idxs = self.data["pos_idxs"] + self.data["neg_idxs"]
            request_idxs = idxs
            idxs = []
            for idx in request_idxs:
                if idx in session_idxs:
                    idxs.append(session_idxs[session_idxs.index(idx)]) # The power of __eq__
                else:
                    idxs.append(ModelIndex(models[self.config["model_name"]] , idx=idx))

        if action == "add_pos" and idxs:
            self.data["pos_idxs"] = list(set(self.data["pos_idxs"]) | set(idxs))  # Union of sets
            self.data["neg_idxs"] = list(set(self.data["neg_idxs"]) - set(idxs))  # Difference of sets

        elif action == "add_neg" and idxs:
            self.data["neg_idxs"] = list(set(self.data["neg_idxs"]) | set(idxs))  # Union of sets
            self.data["pos_idxs"] = list(set(self.data["pos_idxs"]) - set(idxs))  # Difference of sets

        elif action == "remove" and idxs:
            self.data["pos_idxs"] = list(set(self.data["pos_idxs"]) - set(idxs))  # Difference of sets
            self.data["neg_idxs"] = list(set(self.data["neg_idxs"]) - set(idxs))  # Difference of sets

        elif action == "clear":
            self.data["pos_idxs"] = []
            self.data["neg_idxs"] = []
            self.data["res_idxs"] = []

        elif action == "add_upload" and upload:
            self.data["pos_idxs"].append(UploadIndex(upload))

        elif action == "add_prompt" and prompt:
            self.data["pos_idxs"].append(PromptIndex(prompt))
    
    def compute_nns(self):
        nns = models[self.config["model_name"]] .get_nns(emb_type=self.config["emb_type"], metric=self.config["metric"], pos_idxs=self.data["pos_idxs"], neg_idxs=self.data["neg_idxs"], n=self.config["n"])
        # Convert strings to Index-type objects
        self.data["res_idxs"] = [ModelIndex(models[self.config["model_name"]], nn) for nn in nns] # Results can only be model idxs
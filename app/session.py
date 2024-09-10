from app import models, jar
from app.index import ModelIndex, PromptIndex, UploadIndex
from PIL import Image

        
class Session:

    config = {}
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
        self.aliases()

    def aliases(self):
        self.m = models[self.config["model_name"]] 
        self.e = self.config["emb_type"]
        self.t = self.config["metric"]
        self.n = self.config["n"]

    def load_model(self, model_name):
        self.config["private"] = False
        self.config["model_name"] = model_name
        self.config["model_names"] = [model_name for model_name in models.keys() if models[model_name].config["private"] == self.config["private"]]
        self.config["model_len"] = models[self.config["model_name"]].config["model_len"]
        self.config["emb_types"] = list(models[self.config["model_name"]].config["emb_types"].keys())
        self.config["emb_type"] = self.config["emb_types"][0] # ALways put CLIP first in config.json
        self.config["metrics"] = models[self.config["model_name"]].config["emb_types"][self.config["emb_type"]]["metrics"]
        self.config["metric"] = self.config["metrics"][0]
        self.config["ns"] = ["25", "50", "75", "100"]
        self.config["n"] = "50"
        self.aliases()
        
    def edit_idxs(self, action, idxs=None, upload=None, prompt=None):
        assert action in ["add_pos", "add_neg", "remove", "clear", "add_prompt", "add_upload"], f"Invalid action: {action}"
        
        # The form on the client side cannot handle Index-type objects and sends strings so we need to convert them back
        if action == "add_pos" or action == "add_neg" and idxs:
            session_idxs = self.data["pos_idxs"] + self.data["neg_idxs"]
            request_idxs = idxs
            idxs = []
            for idx in request_idxs:
                if idx in session_idxs:
                    idxs.append(session_idxs[session_idxs.index(idx)]) # The power of __eq__
                else:
                    idxs.append(ModelIndex(self.m, idx=idx))

        if action == "add_pos" and idxs:
            self.data["pos_idxs"] = list(set(self.data["pos_idxs"]) | set(idxs))  # Union of sets
            self.data["neg_idxs"] = list(set(self.data["neg_idxs"]) - set(idxs))  # Difference of sets

        elif action == "add_neg" and idxs:
            self.data["neg_idxs"] = list(set(self.data["neg_idxs"]) | set(idxs))  # Union of sets
            self.data["pos_idxs"] = list(set(self.data["pos_idxs"]) - set(idxs))  # Difference of sets

        # This works without alignment through __eq__
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
        # Convert Index-type objects to vectors
        # TODO: This should really happen on the model side but that would to expose the Index class to the model
        pos_idxs = [idx.get_vectors(self.m, self.e, self.t) for idx in self.data["pos_idxs"]]
        neg_idxs = [idx.get_vectors(self.m, self.e, self.t) for idx in self.data["neg_idxs"]]
        # Get NNs
        nns = self.m.get_nns_from_vectors(emb_type=self.e, metric=self.t, pos_idxs=pos_idxs, neg_idxs=neg_idxs, n=self.n)
        # Convert strings to Index-type objects
        self.data["res_idxs"] = [ModelIndex(self.m, nn) for nn in nns] # Results can only be model idxs
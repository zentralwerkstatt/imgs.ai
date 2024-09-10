# Session store to not send everything to the client
# Alternative: Redis etc., e.g. via https://flask-session.readthedocs.io/
# We are running our own as we don't need the scale

import dill as pickle
from uuid import uuid4
import os

class Jar:

    def __init__(self, folder):
        self.new_dir(folder)
        self.folder = folder

    def new_dir(self, folder):
        os.makedirs(folder, exist_ok=True)
        return folder

    def get_id(self):
        return str(uuid4())

    def store(self, data):
        id = self.get_id()
        with open(os.path.join(self.folder, id), "wb") as f:
            pickle.dump(data, f)
        return id

    def restore(self, id):
        with open(os.path.join(self.folder, id), "rb") as f:
            data = pickle.load(f)
        return data
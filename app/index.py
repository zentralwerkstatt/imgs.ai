from uuid import uuid4
from app.util import new_dir
from app.util import CLIP_text


class Index:

    def __init__(self):
        self.vectors = {}
            
    # If we want to hash an object (e.g. insert into a dict) we do it by its idx
    def __hash__(self):
        return hash(self.idx)
    
    # If we want to compare two objects we do it by their idx
    def __eq__(self, other):
        if isinstance(other, str):
            return self.idx == other
        else:
            return self.idx == other.idx
    
    def __str__(self):
        return self.idx
    
class ModelIndex(Index):
    def __init__(self, model, idx):
        super().__init__()
        self.idx = idx
        self.model_name = model.model_name
        metadata = model.get_metadata(self.idx)
        if metadata[0].startswith("http"):
            self.url = metadata[0]
            self.path = None
        else:
            self.url = f"static/{metadata[0]}"
            self.path = metadata[0]
        if len(metadata) > 1:
            source = metadata[1] # Column 2 is source
        else:
            source = "#"
        self.html = f'<img style="width: 100%;" src="{self.url}" />'
        self.modal_body = f'<img style="width: 100%;" src="{self.url}" />'
        self.modal_footer = f'<a href="{source}">Source: {source}</a>'
        # TODO: Implement metadata in footer

    def get_vectors(self, model, emb_type, metric):
        if model.model_name == self.model_name:
            return model.get_vectors_for_idx(self.idx)[emb_type][metric]
        else:
            # TODO: Implement pinned idxs on model change
            return None
    
class UploadIndex(Index):
    def __init__(self, upload):
        super().__init__()
        self.idx = str(uuid4())
        new_dir(f"app/static/user_content")
        self.path = f"app/static/user_content/{self.idx}.jpg"
        upload.save(self.path)
        self.url = f"static/user_content/{self.idx}.jpg"
        self.html = f'<img style="width: 100%;" src="{self.url}" />'
        self.modal_body = f'<img style="width: 100%;" src="{self.url}" />'
        self.modal_footer = ""

    def get_vectors(self, model, emb_type, metric):
        if not model.model_name in self.vectors:
            self.vectors[model.model_name] = model.transform([self.path])
        return self.vectors[model.model_name][emb_type][0] # Images do not have precomputed NNs and thus no metric

class PromptIndex(Index):
    def __init__(self, prompt):
        super().__init__()
        self.idx = str(uuid4())
        self.prompt = prompt
        self.url = None
        self.path = None
        self.vectors = CLIP_text(self.prompt)[0]
        # TODO: Layout for prompt
        self.html = f'<span>{self.prompt}</span>'
        # TODO: Implement modal_body and modal_footer for prompt

    def get_vectors(self, model, emb_type, metric):
        if emb_type.startswith("clip"):
            self.html = f'<span>{self.prompt}</span>'
            return self.vectors
        else:
            # FIXME: Color change does not work
            self.html = f'<span style="color: grey">{self.prompt}</span>'
            return None



from uuid import uuid4
from app.util import new_dir, img_from_url, load_img
from app.util import CLIP_text
import numpy as np
from flask import flash
import typing
from typing import Self
# Dirty hack to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.model import EmbeddingModel
    from PIL import Image


class Index:

    def __init__(self):
        self.vectors = {}

    def keep(self) -> Self:
        return self
            
    # If we want to hash an object (e.g. insert into a dict) we do it by its idx
    def __hash__(self) -> str:
        return hash(self.idx)
    
    # If we want to compare two objects we do it by their idx
    def __eq__(self, other:str | Self) -> bool:
        if isinstance(other, str):
            return self.idx == other
        else:
            return self.idx == other.idx
    
    def __str__(self) -> str:
        return self.idx
    

class UploadIndex(Index):
    def __init__(self, upload:"Image"):
        super().__init__()
        self.idx = str(uuid4())
        new_dir(f"app/static/user_content")
        self.path = f"app/static/user_content/{self.idx}.jpg"
        upload.save(self.path)
        self.url = f"static/user_content/{self.idx}.jpg"
        self.html = f'<img style="width: 100%;" src="{self.url}" />'
        self.modal_body = f'<img style="width: 100%;" src="{self.url}" />'
        self.modal_footer = ""

    def get_vectors(self, model:"EmbeddingModel", emb_type:str, metric:str):
        if not model.model_name in self.vectors:
            self.vectors[model.model_name] = model.transform([self.path])
        return self.vectors[model.model_name][emb_type][0] # Images do not have precomputed NNs and thus no metric

    
class ModelIndex(Index):
    def __init__(self, model:"EmbeddingModel", idx:str):
        super().__init__()
        self.idx = idx
        metadata = model.get_metadata(self)
        if metadata[0].startswith("http"):
            self.url = metadata[0]
            self.path = None
        else:
            self.url = f"static/models/{model.model_name}/data/{metadata[0]}"
            self.path = f"app/static/models/{model.model_name}/data/{metadata[0]}"
        if len(metadata) > 1:
            source = metadata[1] # Column 2 is source
        else:
            source = "#"
        self.html = f'<img style="width: 100%;" src="{self.url}" />'
        self.modal_body = f'<img style="width: 100%;" src="{self.url}" />'
        self.modal_footer = f'<a href="{source}">Source: {source}</a>'
        # TODO: Implement metadata in footer

    def get_vectors(self, model:"EmbeddingModel", emb_type:str, metric:str) -> np.ndarray:
        return model.get_vectors_for_idx(self)[emb_type][metric]
    
    def keep(self) -> UploadIndex:
        img = None
        # Try to get actual image, this could fail for a number of reasons
        try:
            if self.path:
                return UploadIndex(load_img(self.path))
            else:
                return UploadIndex(img_from_url(self.url))
        except:
            flash(f"Could not pin image {self.idx}", "warning")
            return np.zeros(512) # Has to return something


class PromptIndex(Index):
    def __init__(self, prompt):
        super().__init__()
        self.idx = str(uuid4())
        self.prompt = prompt
        self.url = None
        self.path = None
        self.vectors = CLIP_text(self.prompt)[0]
        self.html = f'<div class="text-center" style="min-height: 50px; border: 1px dotted"><span>{self.prompt}</span></div>'
        # TODO: Implement modal_body and modal_footer for prompt

    def get_vectors(self, model:"EmbeddingModel", emb_type:str, metric:str) -> np.ndarray:
        if emb_type.startswith("clip"):
            self.html = f'<div class="text-center" style="min-height: 50px; border: 1px dotted"><span>{self.prompt}</span></div>'
            return self.vectors
        else:
            self.html = f'<div class="text-center" style="min-height: 50px; border: 1px dotted"><span><s>{self.prompt}</s></span></div>'
            return np.zeros(512) # Has to return something
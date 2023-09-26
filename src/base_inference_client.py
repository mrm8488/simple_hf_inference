from abc import ABC
from huggingface_hub import InferenceClient

class BaseInferenceClient(ABC):
    def __init__(self, model: str, token: str, use_cache: bool = True):
        self.client = InferenceClient(model=model, token=token)
        self._set_cache(use_cache)

    def _set_cache(self, use_cache: bool):
        if not use_cache:
            self.client.headers["x-use-cache"] = "0"

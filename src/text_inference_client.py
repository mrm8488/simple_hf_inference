from .base_inference_client import BaseInferenceClient

class TextInferenceClient(BaseInferenceClient):
    def generate_text_response(self, prompt: str, max_new_tokens: int = 200, stream: bool = False):
        return self.client.text_generation(prompt, max_new_tokens=max_new_tokens, stream=stream)

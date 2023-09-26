from .base_inference_client import BaseInferenceClient

class TextToImageInferenceClient(BaseInferenceClient):
    def generate_image_response(self, prompt: str, guidance_scale: int = 9):
        return self.client.text_to_image(prompt, guidance_scale=guidance_scale)

from .text_to_image_inference_client import TextToImageInferenceClient

class StableDiffusionClient(TextToImageInferenceClient):
    def generate_stable_diffusion_response(self, prompt: str, guidance_scale: int = 9):
        return self.generate_image_response(prompt, guidance_scale)

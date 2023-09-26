from .text_inference_client import TextInferenceClient

class CodeModelClient(TextInferenceClient):
    def generate_code_response(self, user_input: str, max_new_tokens: int = 200, stream: bool = False):
        prompt = self._get_code_model_prompt(user_input)
        return self.generate_text_response(prompt, max_new_tokens, stream)

    def _get_code_model_prompt(self, user_input: str) -> str:
        prompt_prefix = 'def remove_non_ascii(s: str) -> str:\n    """ '
        prompt_suffix = "\n    return result"
        return f"<PRE> {prompt_prefix} <SUF>{prompt_suffix} <MID> {user_input}"

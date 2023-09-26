from .text_inference_client import TextInferenceClient

class ChatModelClient(TextInferenceClient):
    def __init__(self, model: str, token: str, use_cache: bool = True):
        super().__init__(model, token, use_cache)
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        return """You are a helpful, respectful, and honest assistant. 
        Always answer as helpfully as possible, while being safe..."""
        # System prompt shortened for brevity, please use the original prompt text.

    def generate_chat_response(self, conversation: list, max_new_tokens: int = 200, stream: bool = False):
        prompt = self._get_chat_model_prompt(conversation)
        return self.generate_text_response(prompt, max_new_tokens, stream)

    def _get_chat_model_prompt(self, conversation: list) -> str:
        user_msgs, model_answers = [], []
        for i, (user_msg, model_answer) in enumerate(conversation):
            user_msgs.append(f"{{ user_msg_{i+1} }} {user_msg}")
            model_answers.append(f"{{ model_answer_{i+1} }} {model_answer}")
        user_msgs_str = " </s><s>[INST] ".join(user_msgs)
        model_answers_str = " ".join(model_answers)
        return f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{user_msgs_str} [/INST] {model_answers_str}"

from chat_model_client import ChatModelClient
from code_model_client import CodeModelClient
from stable_diffusion_client import StableDiffusionClient

token = "YOUR_HUGGINGFACE_API_TOKEN"
chat_model = "meta-llama/Llama-2-70b-chat-hf"

chat_client = ChatModelClient(model=chat_model, token=token, use_cache=False)
chat_conversation = [("How do you make cheese?", "Well, making cheese involves a few key steps...")]

response = chat_client.generate_chat_response(chat_conversation, max_new_tokens=12, stream=True)
print(response)

code_client = CodeModelClient(model="some-code-model", token=token, use_cache=False)
code_response = code_client.generate_code_response("Some user input", max_new_tokens=12, stream=True)
print(code_response)

stable_diffusion_client = StableDiffusionClient(model="some-stable-diffusion-model", token=token, use_cache=False)
image_response = stable_diffusion_client.generate_stable_diffusion_response("Some prompt", guidance_scale=9)
# Process the image response

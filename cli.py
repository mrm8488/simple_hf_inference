import os
import re
import typer
from enum import Enum
from PIL import Image
from src.chat_model_client import ChatModelClient
from src.code_model_client import CodeModelClient
from src.stable_diffusion_client import StableDiffusionClient

class ModelType(Enum):
    CHAT_MODEL = 1
    CODE_MODEL = 2
    STABLE_DIFFUSION_MODEL = 3


class ModelHandler:
    def __init__(self, model, token):
        self.model = model
        self.token = token
        self.activate_model()

    def activate_model(self):
        pass

    def handle(self, user_input):
        pass

class ChatModelHandler(ModelHandler):
    def activate_model(self):
        typer.secho("Chat Model Activated! 💬", bold=True, fg=typer.colors.GREEN)
        
    def handle(self, user_input):
        client = ChatModelClient(model=self.model, token=self.token, use_cache=False)
        conversation = [(user_input, "")]
        response = client.generate_chat_response(conversation, max_new_tokens=200)
        typer.echo(f"{typer.style('Bot:', bold=True, fg=typer.colors.BLUE)} {response} 🤖")

class CodeModelHandler(ModelHandler):
    def activate_model(self):
        typer.secho("Code Model Activated! 💻", bold=True, fg=typer.colors.GREEN)

    def handle(self, user_input):
        client = CodeModelClient(model=self.model, token=self.token, use_cache=False)
        response = client.generate_code_response(user_input)
        typer.echo(f"{typer.style('Bot:', bold=True, fg=typer.colors.BLUE)} {response} 🤖")

class StableDiffusionModelHandler(ModelHandler):
    def activate_model(self):
        typer.secho("Stable Diffusion Model Activated! 🎨", bold=True, fg=typer.colors.GREEN)
        
    def handle(self, user_input):
        client = StableDiffusionClient(model=self.model, token=self.token, use_cache=False)
        response = client.generate_stable_diffusion_response(user_input)
        sanitized_input = re.sub('\W+', '', user_input.replace(' ', '_'))
        file_path = f"{sanitized_input}_generated_image.png"
        response.save(file_path)
        typer.echo(f"{typer.style('Bot:', bold=True, fg=typer.colors.BLUE)} Image saved at {os.path.abspath(file_path)} 🖼️")


app = typer.Typer()

@app.command()
def main(token: str):
    """
    Run the Interactive CLI App for HuggingFace Inference Client 🚀.
    """
    typer.secho("Welcome to HuggingFace Inference Client CLI! 🥳", bold=True, fg=typer.colors.MAGENTA)

    model_types = {
        ModelType.CHAT_MODEL: ("Chat Model 🗣️", "meta-llama/Llama-2-70b-chat-hf", ChatModelHandler),
        ModelType.CODE_MODEL: ("Code Model 💻", "codellama/Llama-2-70b-code-hf", CodeModelHandler),
        ModelType.STABLE_DIFFUSION_MODEL: ("Stable Diffusion Model 🎨", "stabilityai/stable-diffusion-xl-base-1.0", StableDiffusionModelHandler),
    }

    for model_type, (description, _, _) in model_types.items():
        typer.echo(f"{typer.style(str(model_type.value)+'.', bold=True, fg=typer.colors.CYAN)} {description}")

    selected_type = typer.prompt("Enter the number corresponding to the model type")
    if selected_type not in [str(model_type.value) for model_type in ModelType]:
        typer.secho("Invalid Option Selected. 🚫", bold=True, fg=typer.colors.RED)
        return

    selected_type = ModelType(int(selected_type))
    _, default_model, handler_class = model_types[selected_type]
    model = typer.prompt("Enter the model name", default=default_model)

    handler = handler_class(model, token)

    while True:
        user_input = typer.prompt(typer.style("You", bold=True, fg=typer.colors.RED))
        if user_input.lower().strip() == "exit":
            break
        handler.handle(user_input)

if __name__ == "__main__":
    app()

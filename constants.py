from decouple import config

# LLAMA3_URL = config("llama3_url")
HUGGINGFACE_TOKEN = config("huggingface_access_token")
MODEL_PATH = 'models/'
MODEL_NAME = 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'
DEVICE = "cuda"

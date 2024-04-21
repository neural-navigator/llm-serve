"""
This module is responsible to load the model from the memory, if the model is not in memory then this module will
download the model from huggingface
"""
import os
import constants
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, MistralForCausalLM


class SingletonClass(object):
    instance = None

    def __new__(cls):
        if not cls.instance:
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance


class LLMModel(SingletonClass):
    def __init__(self):
        self.model_path = constants.MODEL_PATH
        if os.listdir(self.model_path):
            self.model = MistralForCausalLM.from_pretrained(constants.MODEL_NAME, cache_dir='models/')
            self.tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME, cache_dir='models/')
            # self.model = AutoModel.from_pretrained(self.model_path)
        else:
            print("logging in to the huggingface!")
            login(constants.HUGGINGFACE_TOKEN)
            print("downloading the model!")
            snapshot_download(constants.MODEL_NAME, local_dir='models/')
            # self.model = AutoModel.from_pretrained(self.model_path)
            self.model = MistralForCausalLM.from_pretrained(constants.MODEL_NAME, cache_dir='models/')
            self.tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME, cache_dir='models/')

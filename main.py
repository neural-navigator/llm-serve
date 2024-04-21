from fastapi import FastAPI

import constants
from model import LLMModel
from transformers import AutoTokenizer, MistralForCausalLM
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

app = FastAPI()
llm = LLMModel()

app.model = llm.model


@app.get("/")
def suru_kare():
    return {
        "status_code": 200,
        "message": "Me badhiya! Aap batao!"
    }


@app.get("/jawab_do")
def baat_cheet_karo(user_k_sandesh):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    llm.model.eval()
    llm.model = torch.compile(llm.model, mode="max-autotune", backend="inductor")
    st_time = time.time()
    inputs = llm.tokenizer(user_k_sandesh, return_tensors="pt").to(constants.DEVICE)
    llm.model = llm.model.to(device)
    # Generate
    generate_ids = llm.model.generate(**inputs,
                                      temperature=0.1,
                                      top_k=1,
                                      top_p=1.0,
                                      repetition_penalty=1.4,
                                      min_new_tokens=16,
                                      max_new_tokens=128,
                                      do_sample=True)

    output = llm.tokenizer.decode(generate_ids[0])
    return {
        "status_code": 200,
        "hamara_uttar": output,
        "time_taken": time.time() - st_time
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from src.model import LLMModel
from typing import List
import torch

app = FastAPI()
models_data = {'mistral_ru': ['RefalMachine/llm_model_m_unigram_hm_1e_saiga_v0.1', 'cuda:0']}
models_data = {'mistral_ru': ['IlyaGusev/saiga_mistral_7b_lora', 'cuda:0']}
class SingleGenerationRequest(BaseModel):
    model_name: str
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.2
    repetition_penalty: float = 1.1

class GenerationResponse(BaseModel):
    status: str
    text: str


@app.on_event("startup")
def startup_event():
    global models
    models = {}
    for model_name in models_data:
        models[model_name] = LLMModel(models_data[model_name][0], device=models_data[model_name][1], torch_dtype=torch.float16, load_in_8bit=False)


@app.get("/")
def index():
    return {"text": f"GenerationService: check {app.docs_url} for debug interface and functions"}


@app.post("/generate")
def generate(input: SingleGenerationRequest):
    try:
        text = models[input.model_name].generate(
            input.prompt,
            max_new_tokens=input.max_new_tokens,
            temperature=input.temperature,
            repetition_penalty=input.repetition_penalty
        )
    except Exception as e:
        return GenerationResponse(status=f'ERROR: {str(e)}', translates=[])

    return GenerationResponse(
        status='ok', text=text
    )
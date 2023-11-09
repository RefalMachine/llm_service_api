from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import copy


def generate(
    model,
    tokenizer: AutoTokenizer,
    prompts,
    generation_config: GenerationConfig,
    source_max_length: int = 512,
    eos_token_id: int = None
):
    if eos_token_id is not None:
        generation_config.eos_token_id = eos_token_id

    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=source_max_length,
        padding=True,
        add_special_tokens=False
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )
    outputs = []
    for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
        sample_output_ids = sample_output_ids[len(sample_input_ids):]
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        sample_output = sample_output.replace("</s>", "").strip()
        outputs.append(sample_output)
    return outputs

class LLMModel():
    def __init__(self, model_name, device, torch_dtype, load_in_8bit, use_flash_attention_2=True):
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch_dtype, 
            use_flash_attention_2=use_flash_attention_2, 
            load_in_8bit=load_in_8bit,
            device_map={"": device}
        )

        self.model = PeftModel.from_pretrained(
            model,
            model_name,
            torch_dtype=torch.float16
        )
        
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generation_config = GenerationConfig.from_pretrained(model_name)

    def generate(self, prompt, max_new_tokens=200, temperature=0.2, repetition_penalty=1.0):
        generation_config = copy.deepcopy(self.generation_config)
        generation_config.repetition_penalty = repetition_penalty
        generation_config.max_new_tokens = max_new_tokens
        generation_config.temperature = temperature

        text = generate(self.model, self.tokenizer, [prompt], generation_config)[0]
        return text 

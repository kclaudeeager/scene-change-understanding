import torch
from transformers import  AutoModelForCausalLM

def load_model(model_name="microsoft/Florence-2-large"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype='auto').eval()
    model.to(device)
    return model
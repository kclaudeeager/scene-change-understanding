
from transformers import  AutoProcessor

def load_processor(processor_name="microsoft/Florence-2-large"):
    return AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
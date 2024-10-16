import os
from dotenv import load_dotenv
import openai
from openai import OpenAIError
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv()

def load_llama3_model():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def analyze_changes(roi_history, model_name="gpt-4", instruction_prompt=None):
    alerts = []

    # Default instruction prompt if none is provided
    if instruction_prompt is None:
        instruction_prompt = """Analyze the following scene descriptions over time and identify any significant changes or events that require attention. Focus on changes in objects, people, or activities that might be important for security or monitoring purposes. If there are no significant changes, state that explicitly."""

    if model_name.startswith("gpt"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Make sure it's set in your .env file.")

    for roi_id, descriptions in roi_history.items():
        if len(descriptions) < 2:
            continue
        
        prompt = f"""{instruction_prompt}

Scene descriptions:
{descriptions}

Significant changes or events:"""

        try:
            if model_name.startswith("gpt"):
                response = openai.chat.completions.create(
                    model=model_name, 
                    messages=[
                        {"role": "system", "content": "You are an AI assistant analyzing scene changes for a security monitoring system."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    n=1,
                    stop=None,
                    temperature=0.7,
                )
                analysis = response.choices[0].message.content
            else:
                model, tokenizer = load_llama3_model()
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
                analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"Analysis for ROI {roi_id}: {analysis}")
            
            if "no significant changes" not in analysis.lower():
                alerts.append(f"Alert for ROI {roi_id}: {analysis}")
        
        except (OpenAIError, Exception) as e:
            print(f"An error occurred while analyzing changes: {str(e)}")
    
    return alerts

def send_alerts(alerts):
    for alert in alerts:
        print(f"ALERT: {alert}")
    # Implement your alert system here (e.g., send email, push notification, etc.)
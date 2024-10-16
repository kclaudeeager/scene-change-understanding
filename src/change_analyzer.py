import os
from dotenv import load_dotenv
import openai
from openai import OpenAIError

# Load environment variables
load_dotenv()

def analyze_changes(roi_history, model_name="gpt-4"):
    # Get the API key from the environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai.api_key:
        raise ValueError("OpenAI API key not found. Make sure it's set in your .env file.")

    alerts = []

    for roi_id, descriptions in roi_history.items():
        if len(descriptions) < 2:
            continue
        
        prompt = f"""Analyze the following scene descriptions over time and identify any significant changes or events that require attention. Focus on changes in objects, people, or activities that might be important for security or monitoring purposes. If there are no significant changes, state that explicitly.

Scene descriptions:
{descriptions}

Significant changes or events:"""

        try:
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

            analysis = ""+response.choices[0].message.content
            
            if "no significant changes" not in analysis.lower():
                alerts.append(f"Alert for ROI {roi_id}: {analysis}")
        
        except OpenAIError as e:
            print(f"An error occurred while analyzing changes: {str(e)}")
    
    return alerts

def send_alerts(alerts):
    for alert in alerts:
        print(f"ALERT: {alert}")
        # Implement your alert system here (e.g., send email, push notification, etc.)
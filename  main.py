import yaml
from src.model import load_model
from src.processor import load_processor
from src.roi_tracker import track_rois
from src.change_analyzer import analyze_changes


def load_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    model = load_model(config['model']['name'])
    processor = load_processor(config['model']['name'])
    
    roi_history = track_rois(
        video_source=0,  # Adjust as needed
        model=model,
        processor=processor,
        interval=config['roi_tracker']['interval'],
        duration=config['roi_tracker']['duration']
    )
    
    model_name = config['llm']['model']
    analyze_changes(roi_history, model_name=model_name)

if __name__ == "__main__":
    main()
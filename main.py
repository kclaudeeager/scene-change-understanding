import yaml
import argparse
from src.model import load_model
from src.processor import load_processor
from src.roi_tracker import track_rois
from src.change_analyzer import analyze_changes

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Scene Change Detection with Florence-2")
    parser.add_argument("--video_source", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--interval", type=float, help="Interval between frames (in seconds)")
    parser.add_argument("--duration", type=float, help="Duration of tracking (in seconds)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    model = load_model(config['model']['name'])
    processor = load_processor(config['model']['name'])
    
    # Use command-line arguments if provided, otherwise use config values
    interval = args.interval if args.interval is not None else config['roi_tracker']['interval']
    duration = args.duration if args.duration is not None else config['roi_tracker']['duration']
    
    # Convert video_source to int if it's a digit string (for webcam index)
    video_source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    
    roi_history = track_rois(
        video_source=video_source,
        model=model,
        processor=processor,
        interval=interval,
        duration=duration
    )
    
    print("ROI tracking complete. and returned roi_history: {}".format(roi_history))
    print("Scene tracking complete. Analyzing changes...")
    
    model_name = config['llm']['model']
    analyze_changes(roi_history, model_name=model_name)

if __name__ == "__main__":
    main()
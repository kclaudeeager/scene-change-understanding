import cv2
from PIL import Image
import time
from .inference import analyze_image
from utils import plot_bbox

def track_rois(video_source, model, processor, interval=5, duration=60):
    cap = cv2.VideoCapture(video_source)
    start_time = time.time()
    roi_history = {}

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        result = analyze_image(model, processor, image)
        
        # Assuming the result contains bounding box information
        if 'bboxes' in result and 'labels' in result:
            plot_bbox(image, result)
        
        roi_id = "full_frame"
        if roi_id not in roi_history:
            roi_history[roi_id] = []
        roi_history[roi_id].append(result)

        time.sleep(interval)

    cap.release()
    return roi_history
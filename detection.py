# detection.py

import tensorflow as tf
import numpy as np

# Load the SSD MobileNet model
ssd_model = tf.saved_model.load(r'C:\Users\DELL\OneDrive\Desktop\RANGE\ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model')

def detect_fruits(frame):
    """Detect fruits in the frame using SSD MobileNet."""
    # Preprocess the frame for SSD MobileNet input
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension
    
    # Run object detection
    detections = ssd_model(input_tensor)
    
    # Extract detection boxes, scores, and class indices
    boxes = detections['detection_boxes'][0].numpy()  # Bounding boxes
    scores = detections['detection_scores'][0].numpy()  # Confidence scores
    classes = detections['detection_classes'][0].numpy().astype(int)  # Class IDs
    
    return boxes, scores, classes

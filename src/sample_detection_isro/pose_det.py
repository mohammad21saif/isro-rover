import cv2
import torch
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection

# Load the DETR model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Define a video capture object
vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame
    ret, frame = vid.read()
    if not ret:
        break

    # Preprocess the image
    inputs = processor(images=frame, return_tensors="pt")
    
    # Perform object detection
    outputs = model(**inputs)
    target_sizes = torch.tensor([frame.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.5:
            box = [round(i, 2) for i in box.tolist()]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.config.id2label[label.item()]}: {score:.2f}", 
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow("frame", frame)

    # The 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop, release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

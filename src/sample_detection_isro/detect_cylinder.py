import cv2
from ultralytics import YOLO
import numpy as np

# Load the model
model = YOLO("models/yolov8l-seg.pt")

# Define a video capture object
vid = cv2.VideoCapture(0)

def is_cylinder(contour):
    if len(contour) < 5:
        return False
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        aspect_ratio = ma / (MA + 1e-6)
        if 0.4 < aspect_ratio < 0.6:  # Adjust the ratio based on cylinder properties
            return True
    return False

while True:
    # Capture the video frame
    ret, frame = vid.read()

    if not ret:
        break

    # Make predictions
    results = model.predict(frame)[0]

    if results.masks is None or len(results.masks.data) == 0:
        print("No objects detected")
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Iterate through the segmented objects and draw outlines
    for idx, mask in enumerate(results.masks.data):
        # Convert the mask to a binary image
        mask = mask.cpu().numpy().astype(np.uint8)
        mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            color = (0, 255, 0) if is_cylinder(contour) else (0, 0, 255)

            # Create an RGB version of the mask
            mask_rgb = np.stack([mask * color[2], mask * color[1], mask * color[0]], axis=-1)

            # Blend the mask with the original frame
            frame = cv2.addWeighted(frame, 1, mask_rgb, 0.5, 0)

            # Draw contours on the frame
            cv2.drawContours(frame, [contour], -1, color, 2)

            # Get the class name
            class_name = results.names[results.boxes.cls[idx].item()]

            # Find the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Put the class name near the segmented region
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # The 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop, release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

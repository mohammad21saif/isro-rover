To convert the given OpenCV code into a ROS Noetic node that publishes the segmented cylinder mask on the topic `/sample/points`, you'll need to integrate the OpenCV and YOLOv8 functionalities within a ROS node. Additionally, you will publish the segmented mask as a ROS message. 

Here’s the step-by-step process to achieve this:

### Step 1: Set up the ROS Node
1. Create a new ROS package if you don't already have one:
   ```bash
   catkin_create_pkg vision_node std_msgs sensor_msgs cv_bridge image_transport roscpp rospy
   cd vision_node
   ```

2. Add dependencies to your `CMakeLists.txt` and `package.xml`.

### `CMakeLists.txt`
Add these lines to link the necessary libraries:
```cmake
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
```

### `package.xml`
Include the required dependencies:
```xml
<build_depend>roscpp</build_depend>
<build_depend>rospy</build_depend>
<build_depend>std_msgs</build_depend>
<build_depend>sensor_msgs</build_depend>
<build_depend>cv_bridge</build_depend>
<build_depend>image_transport</build_depend>

<exec_depend>roscpp</exec_depend>
<exec_depend>rospy</exec_depend>
<exec_depend>std_msgs</exec_depend>
<exec_depend>sensor_msgs</exec_depend>
<exec_depend>cv_bridge</exec_depend>
<exec_depend>image_transport</exec_depend>
```

### Step 2: Create the ROS Node
Create a Python script for the ROS node inside the `src` directory of your package.

### `vision_node.py`
Here is the complete code for the ROS node that detects and segments the cylinder:

```python
#!/usr/bin/env python

import rospy
import cv2
from ultralytics import YOLO
import numpy as np
from sentence_transformers import SentenceTransformer, util
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Initialize ROS node
rospy.init_node('vision_node', anonymous=True)
image_pub = rospy.Publisher('/sample/points', Image, queue_size=10)
bridge = CvBridge()

# Load the models
model = YOLO("models/yolov8x-seg.pt")
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the target descriptions and get their embeddings
target_descriptions = ["cylinder", "long object", "cup", "NOT computer"]
target_embeddings = [text_model.encode(description, convert_to_tensor=True) for description in target_descriptions]

def compute_mean_distance(class_name, target_embeddings):
    class_embedding = text_model.encode(class_name, convert_to_tensor=True)
    distances = [1 - util.pytorch_cos_sim(class_embedding, target_embedding).item() for target_embedding in target_embeddings]
    mean_distance = np.mean(distances)
    return mean_distance

def image_callback(data):
    # Convert the ROS image message to OpenCV format
    frame = bridge.imgmsg_to_cv2(data, "bgr8")

    # Make predictions
    results = model.predict(frame)[0]

    if results.masks is None:
        rospy.loginfo("No objects detected")
        return

    closest_class_idx = None
    closest_mean_distance = float('inf')

    # Iterate through the segmented objects and find the closest match based on mean distance
    for idx, mask in enumerate(results.masks.data):
        # Get the class name
        class_name = results.names[results.boxes.cls[idx].item()]

        # Compute the mean distance to the target embeddings
        mean_distance = compute_mean_distance(class_name, target_embeddings)

        if mean_distance < closest_mean_distance:
            closest_mean_distance = mean_distance
            closest_class_idx = idx

    # Draw the segmentation masks
    for idx, mask in enumerate(results.masks.data):
        # Convert the mask to a binary image
        mask = mask.cpu().numpy().astype(np.uint8)
        mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]

        if idx == closest_class_idx:
            color = (0, 255, 0)  # Green for the closest match
            # Convert mask to ROS Image message and publish
            mask_image = bridge.cv2_to_imgmsg(mask, "mono8")
            image_pub.publish(mask_image)
        else:
            color = (0, 0, 255)  # Red for other objects

        # Create an RGB version of the mask
        mask_rgb = np.stack([mask * color[2], mask * color[1], mask * color[0]], axis=-1)

        # Blend the mask with the original frame
        frame = cv2.addWeighted(frame, 1, mask_rgb, 0.5, 0)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the frame
        cv2.drawContours(frame, contours, -1, color, 2)

        # Find the bounding box of the largest contour
        try:
            x, y, w, h = cv2.boundingRect(contours[0])
        except IndexError:
            continue

        # Put the class name and mean distance near the segmented region
        class_name = results.names[results.boxes.cls[idx].item()]
        mean_distance = compute_mean_distance(class_name, target_embeddings)
        text = f"{class_name} ({mean_distance:.2f})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

# Subscribe to the image topic
image_sub = rospy.Subscriber("/camera/image_raw", Image, image_callback)

# Keep the node running
rospy.spin()

# Close OpenCV windows on exit
cv2.destroyAllWindows()
```

### Explanation:
1. **ROS Node Setup**:
   - Initialize the ROS node and create a publisher for the segmented mask image.
   - Subscribe to the camera topic to receive image frames.

2. **Model Loading**:
   - Load the YOLO and SentenceTransformer models.

3. **Target Descriptions**:
   - Compute embeddings for the target descriptions.

4. **Image Callback**:
   - Convert the incoming ROS image to OpenCV format.
   - Perform object detection and segmentation.
   - Compute the mean distance to target embeddings for each detected object.
   - Find the closest match and publish the mask of the detected cylinder.

5. **Run the Node**:
   - The node continuously processes incoming images, detects objects, computes distances, and publishes the segmented mask.

### Step 3: Build and Run the Package
1. **Build the Package**:
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

2. **Run the ROS Node**:
   ```bash
   rosrun vision_node vision_node.py
   ```

Make sure to replace `"/camera/image_raw"` with the appropriate topic name for your camera if it is different. This setup should allow you to detect and segment cylinders in real-time, publishing the segmentation mask to a ROS topic.
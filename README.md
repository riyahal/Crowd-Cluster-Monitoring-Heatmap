# 👥 Crowd & Cluster Monitoring with Heatmap

### <b>This system integrates computer vision, density-based clustering (DBSCAN), and heatmap generation to deliver real-time crowd density analysis and clustering visualization.</b>

<img width="800" height=auto alt="image" src="https://github.com/user-attachments/assets/04ecfdc7-9ab7-4a47-b0fd-4e12260c7fbf" />

## TABLE OF CONTENTS

* <b>[📝 Overview](#overview)</b>

  * [🧩 Core Features](#core-features)
* <b>[📹 Project Demo](#project-demo)</b>
* <b>[⚙️ Technology Stack](#technology-stack)</b>
* <b>[🔄 Workflow](#workflow)</b>
  * [🖧 Diagram](#diagram)
  * [📝 Steps](#steps)

* <b>[📁 Project Structure](#project-structure)</b>
  * [📂 Folder Layout](#folder-layout)
  * [📹 Mobilenet Model Folder](#mobilenet-model-folder)
  * [🧪 Requirements](#requirementstxt)

* <b>[🧠 Code Explanation](#code-explanation)</b>
  * [🐍 crowd_detect.py](#crowd_detectpy)

* <b>[🚀 Steps to Run The Application](#steps-to-run-the-application)</b>
* <b>[✅ Final Notes](#final-notes)</b>
* <b>[License](#license)</b>

---

## Overview

This project utilizes **Python** and **OpenCV** to process video feeds, detect individuals using **MobileNetSSD**, and apply the **DBSCAN algorithm** to identify high-density clusters. It visualizes this data using a color-coded heatmap overlay.

### Core Features

* 📹 **Real-time Video Processing and Person Detection** (OpenCV)
* 🧠 **Deep Learning Detection** (MobileNet SSD)
* 🌡️ **Dynamic Heatmap Generation**
* 📊 **Person Cluster Identification** (DBSCAN)
* 🚨 **Crowd/Cluster Alerts** (Threshold-based warnings)

---

## Project Demo

Here is a screenshot from a run of the project with the bounding boxes around object(s) detected and person and cluster count at the top, as well as heatmap.

<img width="800" height=auto alt="image" src="https://github.com/user-attachments/assets/04ecfdc7-9ab7-4a47-b0fd-4e12260c7fbf" />

---

## Technology Stack

### Core Language & Logic

* **Python** – Script execution and main logic

### Computer Vision

* **OpenCV (cv2)** – Frame capture, preprocessing, drawing functions, heatmap blending

### Machine Learning

* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** – Unsupervised machine learning algorithm in scikit-learn that groups together closely packed data points into dense regions (clusters) and identifies points in low-density areas as noise or outliers.
* **NumPy** – NumPy arrays are the fundamental way images and video frames are stored and accessed in OpenCV's Python bindings. A grayscale image is a 2D array, while a color image is a 3D array (height, width, and color channels, typically BGR). Allows us to perform array and matrix operations such as accessing/modifying pixels, image slicing, cropping, rotation, resizing, and flipping, combining images and adjusting brightness, and drawing lines, rectangles, and text.

---

## Workflow

### Diagram

```text
Camera → MobileNet SSD → Person Detection → Centroids → DBSCAN Clustering → Heatmap Overlay
```

### Steps

1. 📡 Capture webcam feed
2. 🧠 Detect people using MobileNet SSD
3. 👤 Extract centroids (middle of bounding box)
4. 🧩 Cluster using DBSCAN
5. 🎨 Generate heatmap
6. 🖼️ Overlay heatmap on the live frame

---

## Project Structure

### Folder Layout

```text
crowd-monitoring-heatmap-system/
├── mobilenet/                 # MobileNet SSD model files
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── crowd_detect.py            # Main application script
├── requirements.txt           # Dependencies list
├── LICENSE.md                 # License file
└── README.md                  # Documentation (this file)
```

---

### Mobilenet Model Folder

**Contains:**

| File                             | Purpose                                             |
| -------------------------------- | --------------------------------------------------- |
| `MobileNetSSD_deploy.prototxt`   | Defines network architecture (layers, input shape). |
| `MobileNetSSD_deploy.caffemodel` | Pre-trained network weights.                        |

---

### Requirements.txt

Includes:

```text
opencv-python    # 👁️ Used for video processing, drawing, heatmap blending, and MobileNet SSD inference
numpy            # 🔢 Handles array operations, centroid math, and heatmap matrix calculations
scikit-learn     # 📊 Provides DBSCAN clustering for grouping detected people spatially
```

---

## Code Explanation

### `crowd_detect.py`

Below is the complete code and then a breakdown of how each block works.

```python
import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN

# Load model
net = cv2.dnn.readNetFromCaffe(
    'mobilenet/MobileNetSSD_deploy.prototxt',
    'mobilenet/MobileNetSSD_deploy.caffemodel'
)

# Classes (focus only on people)
PERSON_CLASS_ID = 15

# Start webcam
cap = cv2.VideoCapture(0)

ALERT_THRESHOLD = 5
CLUSTER_DISTANCE = 75  # Adjust based on camera scale
CLUSTER_SIZE_THRESHOLD = 3  # Number of people per cluster

ret, frame = cap.read()
h, w = frame.shape[:2]
heatmap = np.zeros((h, w), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    people_centroids = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > 0.5 and class_id == PERSON_CLASS_ID:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if 0 <= cx < w and 0 <= cy < h:
                people_centroids.append([cx, cy])
                heatmap[cy, cx] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 3)

    # Cluster detection
    cluster_count = 0
    if len(people_centroids) > 0:
        people_np = np.array(people_centroids)
        clustering = DBSCAN(eps=CLUSTER_DISTANCE, min_samples=CLUSTER_SIZE_THRESHOLD).fit(people_np)
        labels = clustering.labels_

        unique_clusters = set(labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise points
        cluster_count = len(unique_clusters)

    # Heatmap
    heatmap_blur = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))
    overlay = cv2.addWeighted(heatmap_color, 0.6, frame, 0.4, 0)

    # Display crowd info
    cv2.putText(overlay, f"People: {len(people_centroids)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Clusters: {cluster_count}", (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Optional alert
    if len(people_centroids) >= ALERT_THRESHOLD:
        cv2.putText(overlay, "ALERT: CROWD FORMING", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display
    cv2.namedWindow("CrowdHawk", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CrowdHawk", 1280, 720)
    cv2.imshow("CrowdHawk", overlay)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()```
```

### 🔍 Code Logic Explanation

#### **1. Model Initialization**

* Loads the MobileNet SSD model using:

  ```python
  cv2.dnn.readNetFromCaffe()
  ```
* Only class ID **15** (person) is used.

#### **2. Video Input & Blob Preprocessing**

* Webcam stream:

  ```python
  cap = cv2.VideoCapture(0)
  ```
* Frames are resized and normalized into a **blob** for neural-network inference.

#### **3. Person Detection**

* `net.forward()` returns detections.
* For each detection:

  * Check if `confidence > 0.5`
  * Check `class_id == PERSON_CLASS_ID`
  * Extract centroid `(cx, cy)`
  * Draw bounding box
  * Add intensity point to heatmap

#### **4. DBSCAN Clustering**

* Clusters people based on:

  ```python
  DBSCAN(eps=CLUSTER_DISTANCE, min_samples=CLUSTER_SIZE_THRESHOLD)
  ```
* Removes “noise" points with label `-1`.

#### **5. Heatmap Generation**

* Each detected centroid increments heatmap value.
* Gaussian Blur smooths hotspots.
* Heatmap converted into colorized map via:

  ```python
  cv2.COLORMAP_JET
  ```
* Blended with frame using `cv2.addWeighted`.

#### **6. Alerts & Display**

* Displays:

  * Total people
  * Cluster count
* Shows warning when a crowd is forming
* ESC key exits the program

---

## Steps to Run The Application

### 1. Clone Repository

```bash
git clone https://github.com/rh3nium/Crowd-Cluster-Monitoring-Heatmap
cd Crowd-Cluster-Monitoring-Heatmap
````

### 2. Create & Activate Virtual Environment

```bash
python3 -m venv env
```

**Activate it:**

<i>**Windows:**</i>

```bash
env\Scripts\activate
```

<i>**macOS / Linux:**</i>

```bash
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python3 crowd_detect.py
```
---

## Final Notes

* Increase/reduce **`CLUSTER_DISTANCE`** to change clustering sensitivity.
* Heatmap builds over time → more accurate in dense areas.

---

## License

MIT License. Free to modify and distribute with attribution. Read license here at [LICENSE.md](LICENSE.md).

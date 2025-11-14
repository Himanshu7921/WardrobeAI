# Pose Estimation Module

This directory implements **Model-2** of the WardrobeAI pipeline: **Pose Estimation**.
The purpose of this module is to extract human body landmarks required by downstream virtual try-on components such as the **Geometric Matching Module (GMM)** and the **Try-On Generator**.

MediaPipe Pose is used because it provides lightweight, accurate 33-keypoint detection suitable for single-person fashion try-on systems.

---

## 1. Overview

Pose estimation is required to determine:

* Overall body structure
* Shoulder alignment
* Arm and elbow orientation
* Torso curvature
* Leg positions
* Body-clothing alignment for warping and garment placement

For each input image, this module generates:

1. **A JSON file** containing all 33 landmarks in pixel space
2. **A skeleton visualization image** showing the detected joints and connections

Both outputs are used by later stages of the VTON pipeline.

---

## 2. Model Used

This module uses **MediaPipe Pose (Full-body Landmark Model)** which provides:

* 33 human body landmarks
* Normalized coordinates (x, y, z)
* Visibility score per landmark
* Pose skeleton topology via `POSE_CONNECTIONS`

Since WardrobeAI processes static images, the model is configured with:

```
static_image_mode=True
min_detection_confidence=0.5
min_tracking_confidence=0.5
```

---

## 3. Features

The module performs the following steps:

### 3.1 Pose Estimation

Loads the image and runs MediaPipe Pose:

```python
results, mp_pose, image = get_pose_estimation(image_path)
```

### 3.2 Landmark Extraction

Converts normalized landmarks into absolute pixel coordinates:

* `pixel_landmarks`: used by downstream VTON components
* `normalized_landmarks`: preserved for debugging and flexibility

### 3.3 JSON Generation

Landmarks are saved in a VTON-friendly structure:

```json
{
    "landmarks": [...],
    "num_points": 33,
    "image_size": [height, width]
}
```

### 3.4 Skeleton Rendering

Draws pose connections and joints onto the original image and exports a visual reference.

---

## 4. Directory Structure

```
backend/
└── pose/
    ├── pipeline.py           # Main pose estimation pipeline
    └── README.md             # Documentation
```

Output structure (generated at runtime):

```
data/
└── pose_estimation_output/
    ├── pose_keypoints.json
    └── pose_skeleton.png
```

---

## 5. Running the Pipeline

### Example:

```python
python backend/pose/pipeline.py
```

The script will:

1. Load the input image
2. Detect pose
3. Save keypoints JSON
4. Save skeleton visualization

Ensure paths inside `__main__` are configured properly:

```python
image_path = "path/to/input.png"
save_json_path = "path/to/output/pose_keypoints.json"
save_skeleton_path = "path/to/output/pose_skeleton.png"
```

---

## 6. Output Format

### 6.1 JSON Output

Each landmark contains:

* `x`: pixel coordinate
* `y`: pixel coordinate
* `z`: depth (MediaPipe scale)
* `visibility`: confidence score

### Example Entry:

```json
{
    "landmarks": [
        {"x": 365, "y": 264, "z": -0.76, "visibility": 0.99},
        ...
    ],
    "num_points": 33,
    "image_size": [1080, 720]
}
```

### 6.2 Skeleton Image

A `.png` file containing:

* All landmark joints
* Connections between them
* Visualization based on MediaPipe default drawing styles

---

## 7. Code Structure Summary

### `get_pose_estimation()`

Loads image, initializes MediaPipe, and performs inference.

### `extract_pixel_landmarks()`

Converts normalized coordinates into pixel coordinates.

### `save_pose_json()`

Serializes landmark data to JSON.

### `save_skeleton_image()`

Draws landmarks and pose connections on the image.

### `run_pose_estimation_pipeline()`

Master function combining all steps.

---

## 8. Dependencies

Required packages (matching your backend environment):

```
mediapipe
opencv-python
numpy
```

All dependencies are listed in the root `requirements.txt`.

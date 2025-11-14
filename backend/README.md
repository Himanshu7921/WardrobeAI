## WardrobeAI Backend

The backend component of WardrobeAI provides the complete processing pipeline for virtual try-on functionality, integrating segmentation, pose estimation, garment warping, and try-on generation models into a cohesive system. It exposes these capabilities as modular, replaceable components accessible through a REST API.

---

## Overview

The backend encapsulates all model inference and image-processing logic required to convert a user photo and garment image into a realistic virtual try-on output. It is structured to support individual experimentation, scalable servers, or browser-extension communication.

---

## Core Components

### 1. Segmentation

Implements human parsing, mask visualization, and extraction of task-specific masks.
Directory: `backend/segmentation/`

### 2. Pose Estimation

Estimates human body joints, providing skeleton information required for garment warping.
Directory: `backend/pose/`

### 3. Try-On Generation

Combines segmentation outputs, pose data, and garment inputs to create realistic try-on images using HR-VITON or TryOnDiffusion.
Directory: `backend/tryon/`

### 4. API Interface

REST API that accepts user photos and garment images and returns processed outputs.
Directory: `backend/api/`

### 5. Utilities and Configuration

Logging, path utilities, constants, and shared helper functions.
Directories:

* `backend/config/`
* `backend/utils/`

---

## Processing Pipeline

The end-to-end system completes the following sequence:

1. **Segmentation**
2. **Mask extraction**
3. **Pose estimation**
4. **Garment warping (GMM)**
5. **Try-on generation**
6. **Output rendering**

The pipeline is modular; each step may be replaced or upgraded independently.

---

## Requirements

* Python 3.10
* PyTorch (CUDA-enabled build recommended)
* Transformers
* OpenCV
* Pillow
* NumPy
* FastAPI or Flask (for API layer)

Refer to `requirements.txt` for pinned versions.

---

## Running the Backend

Start the API server:

```
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
```

Segmentation, pose estimation, and try-on can also be executed individually by running the corresponding module scripts.

---

## Notes

* Model weights are not included and must be downloaded separately.
* All personal images and generated results should remain local and are excluded via `.gitignore`.
* The backend is designed for personal use, research experimentation, and extensibility.

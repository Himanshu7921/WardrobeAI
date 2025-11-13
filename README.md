# WardrobeAI

A modular, personal virtual try-on system that generates realistic images of a user wearing any garment from online shopping platforms. WardrobeAI integrates body parsing, pose estimation, garment warping, and generative modeling into a cohesive pipeline. It is designed for personal use, research experimentation, and extensibility.

---

## Overview

WardrobeAI allows a user to upload a single reference photo and virtually try on clothing items such as hoodies, shirts, or jackets. The system extracts product images from e-commerce websites, processes them through a virtual try-on pipeline, and outputs a photorealistic result aligned with the user's pose and body structure.

This repository provides a complete backend pipeline, frontend viewer, and browser extension scaffold.

---

## Features

* Human body parsing for segmenting skin, hair, existing clothing, and other key regions
* Pose estimation for skeleton and body joint detection
* Garment warping using geometric matching
* High-fidelity image generation using a virtual try-on model (HR-VITON or TryOnDiffusion)
* REST API for programmatic virtual try-on requests
* Browser extension that injects a "Try On in WardrobeAI" button on supported product pages
* Modular architecture allowing model replacement or upgrades

---

## Models Used

WardrobeAI uses a three-model architecture to achieve accurate virtual try-on results:

**Human Parsing Model (Model-1): SCHP**
Used to segment body regions such as skin, hair, face, and existing clothing. This enables accurate removal of original garments and precise placement of new clothing items.

**Pose Estimation Model (Model-2): OpenPose**
Used to extract 2D body keypoints including shoulders, elbows, wrists, and torso orientation. This information guides garment alignment and warping based on the user’s posture.

**Try-On Generation Model (Model-3): HR-VITON**
Used to warp the input garment, blend it with the user’s segmented body, and generate a photorealistic final try-on image. HR-VITON provides high-resolution outputs and stable results.

---

## System Architecture

WardrobeAI is structured into three primary components:

### 1. Backend

Implements the full virtual try-on pipeline:

* `segmentation/` handles human parsing
* `pose/` handles pose estimation
* `tryon/` contains cloth warping and generation modules
* `api/` exposes inference endpoints via FastAPI or Flask

### 2. Browser Extension

Injects a client-side button onto product pages, extracts garment images, and communicates with the backend API.

### 3. Frontend

Provides a minimal viewer for rendered try-on results.

---

## Folder Structure

A simplified outline is shown below. For full structure, refer to the repository.

```
WardrobeAI/
  backend/
    segmentation/
    pose/
    tryon/
    api/
    storage/
    config/
    utils/
  browser-extension/
  frontend/
  data/
  experiments/
  scripts/
```

---

## Installation

WardrobeAI requires Python 3.9+ and PyTorch with CUDA for GPU-based inference.

```
cd WardrobeAI
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Install OpenPose, segmentation models, and virtual try-on model weights according to their respective documentation or by using the provided `scripts/download_models.py`.

---

## Running the Backend API

Start the API server:

```
cd backend/api
uvicorn server:app --host 0.0.0.0 --port 8000
```

Test health:

```
GET http://localhost:8000/health
```

---

## Usage Workflow

1. Add one or more static reference photos into `backend/storage/user_photos/`.
2. Use the browser extension on a supported product page to extract the garment image.
3. The extension sends the user image and garment to the backend.
4. WardrobeAI performs segmentation, pose estimation, cloth warping, and generation.
5. The final rendered image is returned to the extension or frontend viewer.

---

## Model Dependencies

WardrobeAI supports multiple model backends:

* Human Parsing: SCHP or any SegFormer-based human parsing model
* Pose Estimation: OpenPose or MediaPipe Pose
* Try-On Generation: HR-VITON, CP-VTON+, or TryOnDiffusion

The system is modular; individual components can be swapped without affecting the API.

---

## Development

All scripts for environment setup, model downloading, and pipeline testing are located in `scripts/`.
Notebooks for experimentation and debugging are available under `experiments/notebooks/`.

---

## Security Notice

This project is designed for personal use. All personal photographs, generated outputs, and garment datasets should remain private and are excluded via `.gitignore` to prevent accidental uploads.

---

## License

This project is licensed under the MIT License.
A copy of the license is provided in the [`LICENSE`](./LICENSE) file.
# Refer experiments/notebooks/model_2.ipynb for experimentations

import cv2
import numpy as np
import json
import mediapipe as mp
import os


def get_pose_estimation(img_path):
    """
    Load image and run MediaPipe Pose model.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,              # For single image inference
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    return results, mp_pose, image


def extract_pixel_landmarks(results, mp_pose, image):
    """
    Convert MediaPipe normalized landmarks → pixel space.
    Returns both pixel_landmarks and normalized_landmarks.
    """

    if not results.pose_landmarks:
        raise ValueError("No pose landmarks detected in the image.")

    image_height, image_width, _ = image.shape

    pixel_landmarks = []
    normalized_landmarks = []

    for lm in results.pose_landmarks.landmark:
        normalized_landmarks.append({
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility
        })

        pixel_landmarks.append({
            'x': int(lm.x * image_width),
            'y': int(lm.y * image_height),
            'z': lm.z,
            'visibility': lm.visibility
        })

    return pixel_landmarks, normalized_landmarks, image_width, image_height

def save_skeleton_image(image, results, mp_pose, save_path):
    """
    Draw pose landmarks & connections on the image and save it.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image_copy = image.copy()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Draw the pose annotation on the image.
    mp_drawing.draw_landmarks(
        image_copy,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    cv2.imwrite(save_path, image_copy)
    print(f"Skeleton image saved to: {save_path}")

def save_pose_json(pixel_landmarks, image_width, image_height, save_path):
    """
    Save pose in VTON-friendly JSON format.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {
        "landmarks": pixel_landmarks,
        "num_points": len(pixel_landmarks),
        "image_size": [image_height, image_width]
    }

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Pose JSON saved to: {save_path}")


def run_pose_estimation_pipeline(image_path, save_json_path, save_skeleton_path):
    results, mp_pose, image = get_pose_estimation(image_path)

    # Extract pixel landmarks
    pixel_landmarks, _, image_width, image_height = extract_pixel_landmarks(
        results, mp_pose, image
    )

    # Save JSON
    save_pose_json(pixel_landmarks, image_width, image_height, save_json_path)

    # Save Skeleton Visualization
    save_skeleton_image(image, results, mp_pose, save_skeleton_path)


if __name__ == "__main__":
    image_path = "D:/Code Playground/wardrob-aI/data/my_photos/person_02.png"

    save_json_path = "D:/Code Playground/wardrob-aI/data/pose_estimation_output/pose_keypoints.json"
    save_skeleton_path = "D:/Code Playground/wardrob-aI/data/pose_estimation_output/pose_skeleton.png"

    run_pose_estimation_pipeline(
        image_path,
        save_json_path,
        save_skeleton_path
    )
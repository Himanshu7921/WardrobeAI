# Refer experiments/notebooks/model_1.ipynb for experimentations

# Importing the necessary libraries for Segmentation and masking task
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from transformers import AutoConfig
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os

# Global variable
ATR_COLORS = {
    0:  (0, 0, 0),          # Background - Black
    1:  (128, 0, 0),        # Hat
    2:  (255, 0, 0),        # Hair
    3:  (255, 255, 0),      # Sunglasses
    4:  (0, 128, 0),        # Upper-clothes
    5:  (0, 255, 0),        # Skirt
    6:  (0, 0, 128),        # Pants
    7:  (0, 0, 255),        # Dress
    8:  (128, 128, 0),      # Belt
    9:  (128, 0, 128),      # Left-shoe
    10: (255, 0, 255),      # Right-shoe
    11: (255, 200, 150),    # Face (skin tone)
    12: (150, 150, 255),    # Left-leg
    13: (180, 180, 255),    # Right-leg
    14: (255, 180, 180),    # Left-arm
    15: (255, 150, 150),    # Right-arm
    16: (0, 150, 150),      # Bag
    17: (0, 255, 255),      # Scarf
}

processor = AutoImageProcessor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
model = AutoModelForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")

# All the pre-processing + input + output of the Model at same time
def human_parse(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    mask = upsampled.argmax(dim=1)[0].numpy()
    return image, mask

def get_id2label():
    config = AutoConfig.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
    id2label = config.id2label
    return id2label

def visualize_mask(mask, save_path="D:/Code Playground/wardrob-ai/wardrob-aI/data/segmentation_output/masks/parsed_output.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)

    for class_id, color in ATR_COLORS.items():
        color_mask[mask == class_id] = color

    Image.fromarray(color_mask).save(save_path)
    print(f"Output mask saved at: {save_path}")

# Extracting all the required Masks
import numpy as np
from PIL import Image

# ATR CLASS LABELS (we can directly fetch this from Hugging-face Model also, 18 classes)
ATR_ID = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper": 4,       # Upper-clothes
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "face": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

# Mask group definations
UPPER_CLOTHES = [ATR_ID["upper"], ATR_ID["dress"]]       # clothing to remove
SKIN = [ATR_ID["face"], ATR_ID["left_arm"], ATR_ID["right_arm"],
        ATR_ID["left_leg"], ATR_ID["right_leg"]]
HAIR = [ATR_ID["hair"]]
BACKGROUND = [ATR_ID["background"]]


def create_mask(pred_mask, class_ids):
    """
    Create a binary mask for given class IDs.
    """
    mask = np.isin(pred_mask, class_ids).astype(np.uint8) * 255
    return mask


def extract_masks(pred_mask, original_image, out_dir="masks"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    """
    Extract and save required WardrobeAI masks:
        - upper clothes
        - skin
        - hair
        - full body (person)
        - agnostic person (clothes removed)
    """

    H, W = pred_mask.shape

    # Convert PIL image → numpy
    img_np = np.array(original_image)

    # MASK: UPPER CLOTHES
    mask_upper = create_mask(pred_mask, UPPER_CLOTHES)
    Image.fromarray(mask_upper).save(f"{out_dir}/mask_upper.png")

    # MASK: SKIN
    mask_skin = create_mask(pred_mask, SKIN)
    Image.fromarray(mask_skin).save(f"{out_dir}/mask_skin.png")

    # 3. MASK: HAIR
    mask_hair = create_mask(pred_mask, HAIR)
    Image.fromarray(mask_hair).save(f"{out_dir}/mask_hair.png")

    # MASK: PERSON / BODY
    all_body_classes = list(range(1, 18))   # everything except background
    mask_body = create_mask(pred_mask, all_body_classes)
    Image.fromarray(mask_body).save(f"{out_dir}/mask_body.png")

    # AGNOSTIC PERSON
    # Start with original
    agnostic = img_np.copy()

    # Remove upper clothes → fill with gray
    agnostic[mask_upper == 255] = [128, 128, 128]

    # Keep skin, face, hair, arms, etc. as is
    # Background remains background

    Image.fromarray(agnostic).save(f"{out_dir}/agnostic_person.png")
    print("All masks saved in:", out_dir)

def run_segmentation_pipeline(image_path):
    image, mask = human_parse(image_path)
    visualize_mask(mask)
    extract_masks(mask, image, out_dir = "D:/Code Playground/wardrob-aI/data/segmentation_output/masks")
if __name__ == "__main__":
    img_path = "D:/Code Playground/wardrob-aI/data/my_photos/person_02.png"
    run_segmentation_pipeline(img_path)
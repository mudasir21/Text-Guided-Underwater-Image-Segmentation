import torch
from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
from PIL import Image
import os

# Function to check file existence
def check_file_exists(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found at {path}")

# Load Grounding DINO model
dino_config = "/home/ghulam/mudasir/dlcv/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
dino_checkpoint = "/home/ghulam/mudasir/dlcv/weights/groundingdino_swint_ogc.pth"
check_file_exists(dino_config, "Grounding DINO config")
check_file_exists(dino_checkpoint, "Grounding DINO checkpoint")
dino_model = load_model(dino_config, dino_checkpoint)

# Load SAM model
sam_checkpoint = "/home/ghulam/mudasir/dlcv/sam_vit_h_4b8939.pth"
check_file_exists(sam_checkpoint, "SAM checkpoint")
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam_predictor = SamPredictor(sam)

# Function to load image from a local path
def load_image(image_path):
    check_file_exists(image_path, "Input image")
    try:
        # image = Image.open(image_path).convert("RGB")
        # return image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_np = cv2.convertScaleAbs(image_np, alpha=1.2, beta=10)  # Enhance contrast
        image = Image.fromarray(image_np)
        return image
    except IOError as e:
        raise RuntimeError(f"Failed to load image from {image_path}: {e}")

# Function to perform detection with Grounding DINO
def detect_objects(image, referring_expression, box_threshold=0.35, text_threshold=0.25):

    image_np = np.array(image)
    # converting rgb to bgr and then to pytorch tensor
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    image_tensor = torch.from_numpy(image_bgr).permute(2,0,1).float()/255.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    image_tensor.to(device)
    # Perform detection
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=referring_expression,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    print(f"Detected {len(boxes)} objects for '{referring_expression}': {phrases}, Logits: {logits}")
    return boxes, image_np



# Function to perform segmentation with SAM
# def segment_objects(image_np, boxes):
#     sam_predictor.set_image(image_np)
#     h, w = image_np.shape[:2]
#     boxes = boxes * torch.tensor([w, h, w, h])
#     masks = []
#     for box in boxes:
#         mask, _, _ = sam_predictor.predict(
#             box=box.numpy(),
#             multimask_output=False
#         )
#         masks.append(mask[0])
#     combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
#     for mask in masks:
#         combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
#     return combined_mask





#      UPDATED 1.1
def segment_objects(image_np, boxes):
    sam_predictor.set_image(image_np)
    h, w = image_np.shape[:2]
    # Convert boxes from normalized (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
    boxes = boxes * torch.tensor([w, h, w, h])
    masks = []
    for i, box in enumerate(boxes):
        # Convert to (x_min, y_min, x_max, y_max)
        x_center, y_center, box_w, box_h = box
        x_min = x_center - box_w / 2
        x_max = x_center + box_w / 2
        y_min = y_center - box_h / 2
        y_max = y_center + box_h / 2
        # Validate box
        if box_w <= 0 or box_h <= 0 or x_min < 0 or x_max > w or y_min < 0 or y_max > h:
            print(f"Skipping invalid box {i}: x_min={x_min:.2f}, y_min={y_min:.2f}, x_max={x_max:.2f}, y_max={y_max:.2f}")
            continue
        # Ensure box coordinates are integers for SAM
        box_array = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        mask, _, _ = sam_predictor.predict(
            box=box_array,
            multimask_output=False
        )
        mask = mask[0]  # Get the first mask
        print(f"Mask {i} for box {box_array}: {mask.sum()} pixels")
        if mask.sum() > 0:  # Only include non-empty masks
            masks.append(mask)
    # Combine masks
    if not masks:
        print("No valid masks generated")
        return np.zeros((h, w), dtype=np.uint8)
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
    print(f"Combined mask: {combined_mask.sum()} pixels")
    return combined_mask

# Function to visualize segmentation
def visualize_segmentation(image, mask, output_path="outputs/segmented_output.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mask = cv2.resize(mask.astype(np.uint8), (image_np.shape[1], image_np.shape[0]))
    colored_mask = np.zeros_like(image_np)
    colored_mask[mask == 1] = [255, 0, 0]  # changed to red in BGR
    output = cv2.addWeighted(image_np, 0.7, colored_mask, 0.8, 0)
    cv2.imwrite(output_path, output)
    return output

# Main function to perform segmentation
def segment_image(image, referring_expression):
    boxes, image_np = detect_objects(image, referring_expression)
    if len(boxes) == 0:
        print(f"No objects detected for: {referring_expression}")
        return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    mask = segment_objects(image_np, boxes)
    return mask

# Main execution
def main(image_path, referring_expressions):
    image = load_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model.to(device)
    sam.to(device)
    
    for expr in referring_expressions:
        print(f"Processing: {expr}")
        output_path = f"outputs/segmented_{expr.replace(' ', '_')}.png"
        mask = segment_image(image, expr)
        output = visualize_segmentation(image, mask, output_path)
        print(f"Output saved as {output_path}")

if __name__ == "__main__":
    # Example usage
    image_path = "/home/ghulam/mudasir/dlcv/inputs/yellow_fish.jpg"  # Replace with your image path
    referring_expressions = [
        "yellow fish"
    ]
    main(image_path, referring_expressions)
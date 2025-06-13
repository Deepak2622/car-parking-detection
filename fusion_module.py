import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Mask R-CNN
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
mask_model = maskrcnn_resnet50_fpn(weights=weights)
mask_model.eval().to(device)
transform = weights.transforms()

# Load YOLOv8 model
yolo_model = YOLO("yolov8m.pt")  # adjust path if needed


def get_fused_predictions(img_path, return_annotated=False):
    """
    Performs fusion between Mask R-CNN and YOLOv8 detections.
    Returns slot-level predictions and optionally, an annotated image.
    """
    img = Image.open(img_path).convert("RGB")
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # === MASK R-CNN prediction ===
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_preds = mask_model(img_tensor)[0]

    masks = mask_preds.get("masks")
    boxes = mask_preds.get("boxes")

    if masks is None or boxes is None:
        return ["empty"], img_cv2 if return_annotated else ["empty"]

    total_slots = masks.shape[0]
    slot_boxes = boxes.cpu().numpy()

    # === YOLOv8 prediction ===
    yolo_results = yolo_model(img_path)
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()

    slot_states = []
    for i, slot_box in enumerate(slot_boxes):
        x1, y1, x2, y2 = slot_box.astype(int)
        slot_center = [(x1 + x2) // 2, (y1 + y2) // 2]

        is_occupied = False
        for vbox in yolo_boxes:
            vx1, vy1, vx2, vy2 = vbox.astype(int)
            if vx1 <= slot_center[0] <= vx2 and vy1 <= slot_center[1] <= vy2:
                is_occupied = True
                break

        slot_states.append("occupied" if is_occupied else "empty")

        # Draw segmentation mask
        mask_np = masks[i][0].mul(255).byte().cpu().numpy()
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        img_cv2[mask_np > 127] = color

    # Blend with YOLO output
    yolo_annotated = yolo_results[0].plot()
    fused_image = cv2.addWeighted(img_cv2, 0.6, yolo_annotated, 0.4, 0)

    if return_annotated:
        return slot_states, fused_image
    return slot_states



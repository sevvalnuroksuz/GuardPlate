import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import easyocr
import re
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

model_path = r"C:\Users\dilek\Desktop\arac_plaka_tanima\best200epoc.pt"
test_images_dir = r"C:\Users\dilek\Desktop\arac_plaka_tanima\dataset\images\test"
test_labels_dir = r"C:\Users\dilek\Desktop\arac_plaka_tanima\dataset\labels\test"
model = YOLO(model_path)
reader = easyocr.Reader(['en'])

iou_threshold = 0.5
y_true, y_pred = [], []
ocr_gt, ocr_pred = [], []

def yolo_to_box(line, iw, ih):
    cls, x, y, w, h = map(float, line.strip().split())
    x1 = int((x - w / 2) * iw)
    y1 = int((y - h / 2) * ih)
    x2 = int((x + w / 2) * iw)
    y2 = int((y + h / 2) * ih)
    return int(cls), [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter_area
    return inter_area / union_area if union_area > 0 else 0

for fname in os.listdir(test_images_dir):
    if not fname.endswith(".jpg"):
        continue

    img_path = os.path.join(test_images_dir, fname)
    label_path = os.path.join(test_labels_dir, fname.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    ih, iw = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()
    gt_boxes, gt_classes, gt_texts = [], [], []
    for line in lines:
        parts = line.strip().split()
        cls, box = yolo_to_box(" ".join(parts[:5]), iw, ih)
        gt_boxes.append(box)
        gt_classes.append(cls)
        gt_texts.append(parts[5].upper() if len(parts) == 6 else "")

    result = model(img)[0]
    pred_boxes = result.boxes.xyxy.cpu().numpy()
    pred_classes = result.boxes.cls.cpu().numpy().astype(int)

    matched = [False] * len(pred_boxes)
    for gt_box, gt_cls, gt_text in zip(gt_boxes, gt_classes, gt_texts):
        found = False
        for i, (pb, pc) in enumerate(zip(pred_boxes, pred_classes)):
            if matched[i]:
                continue
            iou = calculate_iou(gt_box, pb.astype(int).tolist())
            if iou >= iou_threshold:
                y_true.append(gt_cls)
                y_pred.append(pc)
                matched[i] = True
                found = True

                if gt_cls == 0:
                    x1, y1, x2, y2 = map(int, pb)
                    roi = img[y1:y2, x1:x2]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    result = reader.readtext(binary)
                    if result:
                        pred_text = re.sub(r'[^A-Z0-9]', '', result[0][1].upper())
                        ocr_gt.append(gt_text)
                        ocr_pred.append(pred_text)
                break
        if not found:
            y_true.append(gt_cls)
            y_pred.append(-1)

ocr_scores = [SequenceMatcher(None, a, b).ratio() for a, b in zip(ocr_gt, ocr_pred)]

precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
ocr_acc = np.mean(ocr_scores) if ocr_scores else 0

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"OCR Accuracy (Similarity): {ocr_acc:.3f}")

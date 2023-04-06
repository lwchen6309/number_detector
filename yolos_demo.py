from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image, ImageDraw
import cv2
import torch
import torchvision.ops as ops


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# Read the video file
cap = cv2.VideoCapture('VID_20230131_103150_1_trunc.mp4')

# Extract the first frame
_, image = cap.read()
# Convert the image from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
# image_pil = Image.open('./dog-puppy-on-garden-royalty-free-image-1586966191.jpeg')

model_id = 'hustvl/yolos-small'
feature_extractor = YolosFeatureExtractor.from_pretrained(model_id)
model = YolosForObjectDetection.from_pretrained(model_id)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = model.to(device)

inputs = feature_extractor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
# img_shape = inputs['pixel_values'].shape[-2:]

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes

# decode the boxes in YOLO format
bboxes = box_cxcywh_to_xyxy(bboxes)[0]
# convert scores to probabilities
scores = logits.softmax(-1)[0]

# Detect the position of laptop
laptop_class = model.config.label2id['laptop']
scores = scores[:,laptop_class]

# perform NMS to filter out overlapping boxes
box_idx = ops.nms(bboxes, scores, iou_threshold=0.5)
box_idx = box_idx[scores[box_idx] > 0.3]
nms_boxes = bboxes[box_idx]
nms_scores = scores[box_idx]

# Unnormalize the box coordinates and plot the boxes and scores on the image
draw = ImageDraw.Draw(image)
for box, score in zip(nms_boxes, nms_scores):
    # Unnormalize the box coordinates
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x_center, y_center = x1 + w / 2, y1 + h / 2
    x1, y1, x2, y2 = x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height

    # Draw the box and score on the image
    draw.rectangle((x1, y1, x2, y2), outline ="red", width=2)
    try:
        text = '%s: %.2f'%(model.config.id2label[laptop_class], float(score))
    except KeyError:
        text = '%s: %.2f'%('Unknown', float(score))
    draw.text((x_center - 10, y_center - 10), text, fill="red")
image.show()
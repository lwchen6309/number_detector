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

def build_model():
    model_id = 'hustvl/yolos-small'
    feature_extractor = YolosFeatureExtractor.from_pretrained(model_id)
    model = YolosForObjectDetection.from_pretrained(model_id)
    return feature_extractor, model

# add iou_threshold and score_threshold to run_detector as arguments
def run_detector(image, feature_extractor, model, iou_threshold=0.5, score_threshold=0.5):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # decode the boxes in YOLO format
    bboxes = box_cxcywh_to_xyxy(bboxes)[0]
    # convert scores to probabilities
    prob = logits.softmax(-1)[0]
    scores, labels = prob[..., :-1].max(-1)

    # perform NMS to filter out overlapping boxes
    box_idx = ops.nms(bboxes, scores, iou_threshold=iou_threshold)
    box_idx = box_idx[scores[box_idx] > score_threshold]
    nms_boxes = bboxes[box_idx]
    nms_scores = scores[box_idx]
    labels = labels[box_idx]
    return nms_boxes, nms_scores, labels

def detect_laptop(image):
    feature_extractor, model = build_model()
    nms_boxes, nms_scores, labels = run_detector(image, feature_extractor, model, score_threshold=0.5)

    # Detect the position of laptop
    laptop_class = model.config.label2id['laptop']

    is_target = labels == laptop_class
    nms_boxes = nms_boxes[is_target]
    nms_scores = nms_scores[is_target]
    labels = labels[is_target]
    labels = [model.config.id2label[int(label)] for label in labels]
    # Unnormalize the box coordinates and plot the boxes and scores on the image
    nms_boxes = nms_boxes * torch.tensor([image.width, image.height, image.width, image.height])
    return nms_boxes, nms_scores, labels


if __name__ == '__main__':
    # Read the video file
    cap = cv2.VideoCapture('VID_20230131_103150_1_trunc.mp4')

    # Extract the first frame
    _, image = cap.read()

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    nms_boxes, nms_scores, labels = detect_laptop(image)

    draw = ImageDraw.Draw(image)
    for box, score, label in zip(nms_boxes, nms_scores, labels):
        # Unnormalize the box coordinates
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        x_center, y_center = x1 + w / 2, y1 + h / 2

        # Draw the box and score on the image
        draw.rectangle((x1, y1, x2, y2), outline ="red", width=2)
        text = '%s: %.2f'%(label, float(score))
        draw.text((x_center - 10, y_center - 10), text, fill="red")
    image.show()
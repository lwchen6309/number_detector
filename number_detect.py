from transformers import YolosFeatureExtractor, YolosForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torch
import torchvision.ops as ops
import os
from tqdm import tqdm


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def build_object_detection_model():
    model_id = 'hustvl/yolos-small'
    feature_extractor = YolosFeatureExtractor.from_pretrained(model_id)
    model = YolosForObjectDetection.from_pretrained(model_id)
    return feature_extractor, model

def build_ocr_model():
    model_id = 'microsoft/trocr-base-str'
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model = model.to('cuda')
    return processor, model

# add iou_threshold and score_threshold to run_detector as arguments
def run_object_detector(image, feature_extractor, model, iou_threshold=0.5, score_threshold=0.5):
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

def detect_laptop(image, object_feature_extractor, object_model):
    nms_boxes, nms_scores, labels = run_object_detector(image, object_feature_extractor, object_model, score_threshold=0.5)

    # Detect the position of laptop
    laptop_class = object_model.config.label2id['laptop']

    is_target = labels == laptop_class
    nms_boxes = nms_boxes[is_target]
    nms_scores = nms_scores[is_target]
    labels = labels[is_target]
    labels = [object_model.config.id2label[int(label)] for label in labels]
    # Unnormalize the box coordinates and plot the boxes and scores on the image
    nms_boxes = nms_boxes * torch.tensor([image.width, image.height, image.width, image.height])
    return nms_boxes, nms_scores, labels


def detect_text_in_frames(cap, model, processor, bbox_list):
    # Loop through the frames and extract the text for each bounding box
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    result = []
    for i in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if ret == False:
            break
        # Check if the current frame index is a multiple of the frame rate
        if i % fps == 0:
            # Loop through each bounding box and extract the text
            for bbox_idx, bbox in enumerate(bbox_list):
                # Check if the current frame is within the time range of the bounding box
                if i >= bbox[0] and i <= bbox[1]:
                    # Extract the region of the image within the bounding box
                    x1, y1, x2, y2 = bbox[2]
                    region = frame[int(y1):int(y2), int(x1):int(x2), :]
                    # Convert the region to grayscale
                    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    # Convert the grayscale region to PIL image
                    image = Image.fromarray(gray_region).convert("RGB")
                    # Preprocess the image for OCR
                    pixel_values = processor(image, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to('cuda')
                    # Generate the text from the image using the OCR model
                    generated_ids = model.generate(pixel_values, pad_token_id=2)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    result.append((bbox_idx, i, generated_text))
    return result

if __name__ == '__main__':
    # Read the video file
    cap = cv2.VideoCapture('VID_20230131_103150_1_trunc.mp4')

    # Extract the first frame
    _, image = cap.read()

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Detect the position of laptop
    object_feature_extractor, object_model = build_object_detection_model()
    nms_boxes, nms_scores, labels = detect_laptop(image, object_feature_extractor, object_model)

    # Convert the bounding boxes to (start_frame, end_frame, box) format
    bbox_list = []
    for box in nms_boxes:
        x1, y1, x2, y2 = box
        start_frame = 0
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bbox_list.append((start_frame, end_frame, (x1, y1, x2, y2)))

    # Detect text in frames for each bounding box
    ocr_processor, ocr_model = build_ocr_model()
    basedir = './VID_20230131_103150_1_trunc'
    subdirs = sorted([os.path.join(basedir,f) for f in os.listdir(basedir) 
                      if os.path.isdir(os.path.join(basedir,f)) and f.startswith("region_")])
    for subdir in subdirs:
        # Get the full path of the video file in the subdirectory
        video_path = os.path.join(subdir, "output.mp4")
        # Check if the video file exists
        if not os.path.exists(video_path):
            continue
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Detect text in frames for each bounding box
        result = detect_text_in_frames(cap, ocr_model, ocr_processor, bbox_list)

        # Close the video file
        cap.release()
        # Write the result to a text file in the same directory as the video file
        result_path = os.path.join(subdir, "result.txt")
        with open(result_path, "w") as f:
            for bbox_idx, frame_idx, text in result:
                f.write('%d, %d, %s\n'%(bbox_idx, frame_idx, text))
       

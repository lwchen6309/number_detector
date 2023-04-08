import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import easyocr


def estimate_gradient_threshold(cap):
    # Loop through the first num_grad_frames frames and calculate the gradient along the time axis
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame = None
    gradients = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate the gradient along the time axis
        if prev_frame is not None:
            gradient = cv2.absdiff(gray_frame, prev_frame).sum()
            gradients.append(gradient)
        # Set the current frame as the previous frame for the next iteration
        prev_frame = gray_frame
    # Smooth the gradients using a moving average filter with window size 5
    filtered_gradients = np.convolve(gradients, np.ones(10)/10, mode='valid')
    # filtered_gradients = gradients
    # Calculate the median gradient and set the gradient threshold to 3 times the median
    grad_threshold = 5 * np.median(filtered_gradients)
    return filtered_gradients, grad_threshold

def build_trocr_model():
    # Load the TrOCR model
    model_id = 'microsoft/trocr-base-str'
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    return model, processor

def build_easyocr_model():
    reader = easyocr.Reader(['ch_tra','en'], gpu=False)
    return reader

def read_video_trocr(cap, model, processor, bbox, time_step_in_second = 10, device='cpu'):
    # Get the coordinates of the bounding box
    x1, y1, x2, y2 = bbox
    model = model.to(device)
    # Open the video file
    # Get the number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Infer once per 10 seconds of video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_step = int(time_step_in_second * frame_rate)
    num_steps = int(num_frames / frame_step) + 1
    result = []
    for i in tqdm(range(num_steps)):
        start_frame = i * frame_step
        end_frame = min(num_frames, start_frame + frame_step)
        if start_frame >= end_frame:
            break
        # Set the video file position to the current frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if ret == False:
            break
        image = Image.fromarray(frame[y1:y2, x1:x2, :]).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        generated_ids = model.generate(pixel_values, pad_token_id=2)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result.append((start_frame / frame_rate, generated_text))
    return result

def read_video_easyocr(cap, reader, bbox, time_step_in_second = 10):
    # Get the coordinates of the bounding box
    x1, y1, x2, y2 = bbox
    # Open the video file
    # Get the number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Infer once per 10 seconds of video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_step = int(time_step_in_second * frame_rate)
    num_steps = int(num_frames / frame_step) + 1
    result = []
    for i in tqdm(range(num_steps)):
        start_frame = i * frame_step
        end_frame = min(num_frames, start_frame + frame_step)
        if start_frame >= end_frame:
            break
        # Set the video file position to the current frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if ret == False:
            break
        image = frame[y1:y2, x1:x2, :]
        generated_text = reader.readtext(image)
        if len(generated_text) > 0:
            # generated_text = generated_text[0][1]
            result.append((start_frame / frame_rate, generated_text))
    return result


if __name__ == '__main__':
    # model, processor = build_trocr_model()
    reader = easyocr.Reader(['ch_tra','en'], gpu=False)
    # Get a list of all subdirectories that start with "region_"
    video_path = './slice_1.mp4'
    # video_path = './VID_20230131_103150_1_trunc.mp4'
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the size of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bbox = (0,0,width,height)
    # Get the number of frames in the video
    # result = read_video_trocr(cap, model, processor, bbox, device='mps')
    result = read_video_easyocr(cap, reader, bbox, time_step_in_second=10)
    # Close the video file
    cap.release()
    basedir = '.'
    # Write the result to a text file in the same directory as the video file
    result_path = os.path.join(basedir, "result.txt")
    with open(result_path, "w") as f:
        for res in result:
            f.write('%s, %s\n' % (res[0], res[1]))
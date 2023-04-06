import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


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


model_id = 'microsoft/trocr-base-str'
processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)
model = model.to('cuda')

# Get a list of all subdirectories that start with "region_"
basedir = './VID_20230131_103150_1_trunc'
subdirs = sorted([os.path.join(basedir,f) for f in os.listdir(basedir) 
                  if os.path.isdir(os.path.join(basedir,f)) and f.startswith("region_")])

# Loop through each subdirectory and extract the text for each frame in that subdirectory
for subdir in subdirs:
    # Get the full path of the video file in the subdirectory
    video_path = os.path.join(subdir, "output.mp4")
    # Check if the video file exists
    if not os.path.exists(video_path):
        continue
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    filtered_gradients, grad_threshold = estimate_gradient_threshold(cap)
    idx = np.where(filtered_gradients > grad_threshold)[0]
    
    # Loop through each frame and extract the text
    result = []
    grads = []
    prev_frame = None
    for i in tqdm(idx):
        # Set the video file position to the current frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret == False:
            break
        image = Image.fromarray(frame).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to('cuda')
        generated_ids = model.generate(pixel_values, pad_token_id=2)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result.append(generated_text)
    
    # Close the video file
    cap.release()
    # Write the result to a text file in the same directory as the video file
    result_path = os.path.join(subdir, "result.txt")
    with open(result_path, "w") as f:
        for i, text in zip(idx, result):
            f.write('%s, %s\n'%(i, text))

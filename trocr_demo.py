import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
from PIL import Image


model_id = 'microsoft/trocr-base-str'
processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)

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
    # Loop through each frame and extract the text
    result = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        # Perform OCR inference once every 10 frames
        if i % 10 == 0:
            image = Image.fromarray(frame).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            result.append(generated_text)
        # Print the number of frames processed
        print("Processed frame {} of {}".format(i+1, num_frames))
    # Close the video file
    cap.release()
    # Write the result to a text file in the same directory as the video file
    result_path = os.path.join(subdir, "result.txt")
    with open(result_path, "w") as f:
        for text in result:
            f.write(text + "\n")

import cv2


# Set the start and end times (in seconds)
start_time = 732  # 12:12 in seconds
end_time = 735  # 26:14 in seconds

# Open the video file
cap = cv2.VideoCapture('./VID_20230131_103150_1.mp4')

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the start and end frames based on the start and end times
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Set the current frame to the start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Read frames from the video and save them as JPEG image files
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save the frame as a JPEG image file if we're in the specified slice
    if frame_count >= start_frame and frame_count <= end_frame:
        cv2.imwrite(f'frame_{frame_count}.jpg', frame)
    
    # Exit the loop if we've reached the end frame
    if frame_count == end_frame:
        break
    
    frame_count += 1

# Release the video file
cap.release()

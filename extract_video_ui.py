import cv2
import os
from tkinter import *
from tkinter import filedialog

# Set up the UI
root = Tk()
root.title("Video Region Extractor")
root.geometry("400x400")

# Select the video file
def select_file():
    file_path = filedialog.askopenfilename()
    file_path_entry.delete(0, END)
    file_path_entry.insert(0, file_path)

# Create the output directories
def create_directories():
    video_path = file_path_entry.get()
    output_path = os.path.splitext(video_path)[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(len(regions)):
        region_path = os.path.join(output_path, "region_" + str(i))
        if not os.path.exists(region_path):
            os.makedirs(region_path)

# Extract the selected regions from the video and save as separate video files
def extract_regions():
    video_path = file_path_entry.get()
    cap = cv2.VideoCapture(video_path)

    # Create video writer objects for all selected regions
    output_path = os.path.splitext(video_path)[0]
    writers = []
    for j, (x, y, w, h) in enumerate(regions):
        region_path = os.path.join(output_path, "region_" + str(j), "output.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(region_path, fourcc, fps, (w, h))
        writers.append(writer)

    # Process each frame of the video and write to the appropriate video writer
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        for j, (x, y, w, h) in enumerate(regions):
            region_frame = frame[y:y+h, x:x+w]
            writers[j].write(region_frame)

    # Release video capture and writer objects
    cap.release()
    for writer in writers:
        writer.release()



# Set up the file path entry
file_path_label = Label(root, text="Video File:")
file_path_label.pack()
file_path_entry = Entry(root, width=50)
file_path_entry.pack()
select_file_button = Button(root, text="Select", command=select_file)
select_file_button.pack()

# Set up the region selection buttons
regions = []
def select_region():
    global regions
    cap = cv2.VideoCapture(file_path_entry.get())
    ret, frame = cap.read()
    cv2.namedWindow("Select Region")
    cv2.imshow("Select Region", frame)
    rect = cv2.selectROI("Select Region", frame, fromCenter=False, showCrosshair=True)
    regions.append(rect)
    cv2.destroyWindow("Select Region")
    cap.release()

select_region_button = Button(root, text="Select Region", command=select_region)
select_region_button.pack()

# Set up the extraction button
extract_button = Button(root, text="Extract Regions", command=lambda:[create_directories(), extract_regions()])
extract_button.pack()

root.mainloop()
